"""
Statistical Analysis for Dynamic Pricing

This module provides statistical methods for:
- Price elasticity estimation with confidence intervals
- A/B testing with sequential analysis and early stopping
- Bayesian price optimization
- Causal inference for pricing effects

These methods ensure pricing decisions are statistically sound
and properly account for uncertainty.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.special import gammaln
import logging

logger = logging.getLogger(__name__)


# ============== Price Elasticity Estimation ==============

@dataclass
class ElasticityEstimate:
    """Price elasticity estimation result."""
    point_estimate: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    r_squared: float
    n_observations: int
    method: str


def estimate_elasticity_ols(
    log_prices: np.ndarray,
    log_quantities: np.ndarray,
    controls: Optional[np.ndarray] = None,
    confidence_level: float = 0.95
) -> ElasticityEstimate:
    """
    Estimate price elasticity using OLS regression.
    
    Model: log(Q) = α + ε*log(P) + β*X + error
    
    Args:
        log_prices: Log of prices (n_obs,)
        log_quantities: Log of quantities (n_obs,)
        controls: Control variables (n_obs, n_controls)
        confidence_level: Confidence level for interval
        
    Returns:
        ElasticityEstimate with point estimate and uncertainty
    """
    n = len(log_prices)
    
    # Build design matrix
    if controls is not None:
        X = np.column_stack([np.ones(n), log_prices, controls])
    else:
        X = np.column_stack([np.ones(n), log_prices])
    
    y = log_quantities
    
    # OLS estimation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    # Residuals and variance
    residuals = y - X @ beta
    sigma2 = np.sum(residuals**2) / (n - X.shape[1])
    
    # Standard errors
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    
    # Elasticity is second coefficient
    elasticity = beta[1]
    se_elasticity = se_beta[1]
    
    # T-statistic and p-value
    t_stat = elasticity / se_elasticity
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1]))
    
    # Confidence interval
    t_crit = stats.t.ppf((1 + confidence_level) / 2, n - X.shape[1])
    ci_lower = elasticity - t_crit * se_elasticity
    ci_upper = elasticity + t_crit * se_elasticity
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot
    
    return ElasticityEstimate(
        point_estimate=elasticity,
        std_error=se_elasticity,
        confidence_interval=(ci_lower, ci_upper),
        p_value=p_value,
        r_squared=r_squared,
        n_observations=n,
        method="OLS"
    )


def estimate_elasticity_iv(
    log_prices: np.ndarray,
    log_quantities: np.ndarray,
    instruments: np.ndarray,
    controls: Optional[np.ndarray] = None,
    confidence_level: float = 0.95
) -> ElasticityEstimate:
    """
    Estimate price elasticity using Instrumental Variables (2SLS).
    
    Use this when prices are endogenous (correlated with demand shocks).
    Common instruments: cost shifters, competitor prices, weather.
    
    Args:
        log_prices: Log of prices (n_obs,)
        log_quantities: Log of quantities (n_obs,)
        instruments: Instrumental variables (n_obs, n_instruments)
        controls: Control variables
        confidence_level: Confidence level
        
    Returns:
        ElasticityEstimate
    """
    n = len(log_prices)
    
    # Build matrices
    if controls is not None:
        exog = np.column_stack([np.ones(n), controls])
        Z = np.column_stack([exog, instruments])
    else:
        exog = np.ones((n, 1))
        Z = np.column_stack([exog, instruments])
    
    # First stage: regress price on instruments
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    gamma = ZtZ_inv @ Z.T @ log_prices
    price_hat = Z @ gamma
    
    # First stage F-statistic (instrument strength)
    ss_first = np.sum((price_hat - np.mean(log_prices))**2)
    ss_resid = np.sum((log_prices - price_hat)**2)
    f_stat = (ss_first / len(instruments)) / (ss_resid / (n - Z.shape[1]))
    
    if f_stat < 10:
        logger.warning(f"Weak instruments: F-statistic = {f_stat:.2f} < 10")
    
    # Second stage: regress quantity on predicted price
    X_second = np.column_stack([exog, price_hat])
    XtX_inv = np.linalg.inv(X_second.T @ X_second)
    beta = XtX_inv @ X_second.T @ log_quantities
    
    # Correct standard errors (use actual residuals)
    residuals = log_quantities - np.column_stack([exog, log_prices]) @ beta
    sigma2 = np.sum(residuals**2) / (n - X_second.shape[1])
    
    # Variance calculation for 2SLS
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    
    elasticity = beta[-1]  # Last coefficient is price
    se_elasticity = se_beta[-1]
    
    # Confidence interval
    t_crit = stats.t.ppf((1 + confidence_level) / 2, n - X_second.shape[1])
    ci_lower = elasticity - t_crit * se_elasticity
    ci_upper = elasticity + t_crit * se_elasticity
    
    # P-value
    t_stat = elasticity / se_elasticity
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - X_second.shape[1]))
    
    # R-squared (pseudo for IV)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_quantities - np.mean(log_quantities))**2)
    r_squared = 1 - ss_res / ss_tot
    
    return ElasticityEstimate(
        point_estimate=elasticity,
        std_error=se_elasticity,
        confidence_interval=(ci_lower, ci_upper),
        p_value=p_value,
        r_squared=r_squared,
        n_observations=n,
        method=f"2SLS (F-stat: {f_stat:.1f})"
    )


# ============== A/B Testing Framework ==============

@dataclass
class ABTestResult:
    """Result of A/B test analysis."""
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float  # Lift
    confidence_interval: Tuple[float, float]
    p_value: float
    power: float
    significant: bool
    sample_size_control: int
    sample_size_treatment: int
    recommendation: str


class SequentialABTest:
    """
    Sequential A/B testing with early stopping.
    
    Uses group sequential design to allow for early termination
    while controlling Type I and Type II error rates.
    
    Features:
    - O'Brien-Fleming spending function for α-spending
    - Futility bounds for early stopping
    - Proper multiple testing correction
    
    Example:
        test = SequentialABTest(
            alpha=0.05,
            power=0.80,
            minimum_effect=0.05,
            max_looks=5
        )
        
        # At each interim analysis
        result = test.analyze(control_data, treatment_data)
        if result.significant or test.should_stop_for_futility():
            # Stop test
            pass
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        minimum_detectable_effect: float = 0.05,
        max_looks: int = 5,
        spending_function: str = "obrien_fleming"
    ):
        self.alpha = alpha
        self.power = power
        self.mde = minimum_detectable_effect
        self.max_looks = max_looks
        self.spending_function = spending_function
        
        # Track analysis history
        self.current_look = 0
        self.analysis_history: List[ABTestResult] = []
        
        # Compute spending boundaries
        self.alpha_spent = 0.0
        self.boundaries = self._compute_boundaries()
    
    def _compute_boundaries(self) -> List[float]:
        """Compute critical values at each look using O'Brien-Fleming."""
        boundaries = []
        
        for k in range(1, self.max_looks + 1):
            information_fraction = k / self.max_looks
            
            if self.spending_function == "obrien_fleming":
                # O'Brien-Fleming: very conservative early, liberal late
                alpha_k = 2 * (1 - stats.norm.cdf(
                    stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(information_fraction)
                ))
            else:
                # Pocock: equal α-spending at each look
                alpha_k = self.alpha / self.max_looks
            
            z_crit = stats.norm.ppf(1 - alpha_k / 2)
            boundaries.append(z_crit)
        
        return boundaries
    
    def analyze(
        self,
        control_revenue: np.ndarray,
        treatment_revenue: np.ndarray
    ) -> ABTestResult:
        """
        Perform interim analysis.
        
        Args:
            control_revenue: Revenue per user in control
            treatment_revenue: Revenue per user in treatment
            
        Returns:
            ABTestResult with current test status
        """
        self.current_look += 1
        
        n_control = len(control_revenue)
        n_treatment = len(treatment_revenue)
        
        # Compute means and variances
        mean_control = np.mean(control_revenue)
        mean_treatment = np.mean(treatment_revenue)
        var_control = np.var(control_revenue, ddof=1)
        var_treatment = np.var(treatment_revenue, ddof=1)
        
        # Effect estimates
        absolute_effect = mean_treatment - mean_control
        relative_effect = absolute_effect / mean_control if mean_control > 0 else 0
        
        # Pooled standard error
        se = np.sqrt(var_control / n_control + var_treatment / n_treatment)
        
        # Test statistic
        z_stat = absolute_effect / se
        
        # Get critical value for this look
        if self.current_look <= len(self.boundaries):
            z_crit = self.boundaries[self.current_look - 1]
        else:
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
        
        # P-value (unadjusted)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Significance using sequential boundary
        significant = abs(z_stat) > z_crit
        
        # Confidence interval
        ci_lower = absolute_effect - z_crit * se
        ci_upper = absolute_effect + z_crit * se
        
        # Power calculation
        effect_under_alternative = mean_control * self.mde
        ncp = effect_under_alternative / se  # Non-centrality parameter
        power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        
        # Recommendation
        if significant:
            if relative_effect > 0:
                recommendation = f"STOP: Treatment wins with {relative_effect*100:.1f}% lift"
            else:
                recommendation = f"STOP: Control wins, treatment {relative_effect*100:.1f}% worse"
        elif self.current_look >= self.max_looks:
            recommendation = "CONCLUDE: No significant difference detected"
        else:
            recommendation = f"CONTINUE: Look {self.current_look}/{self.max_looks}"
        
        result = ABTestResult(
            control_mean=mean_control,
            treatment_mean=mean_treatment,
            absolute_effect=absolute_effect,
            relative_effect=relative_effect,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            power=power,
            significant=significant,
            sample_size_control=n_control,
            sample_size_treatment=n_treatment,
            recommendation=recommendation
        )
        
        self.analysis_history.append(result)
        return result
    
    def should_stop_for_futility(self, futility_threshold: float = 0.20) -> bool:
        """
        Check if test should stop for futility.
        
        Stop if conditional power is too low to detect effect.
        """
        if not self.analysis_history:
            return False
        
        latest = self.analysis_history[-1]
        return latest.power < futility_threshold
    
    def required_sample_size(self) -> int:
        """Compute required sample size per group."""
        # Two-sample z-test sample size
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)
        
        # Assuming equal variances and means around 100
        assumed_mean = 100
        assumed_std = 30  # CV = 30%
        
        effect_size = assumed_mean * self.mde
        
        n = 2 * ((z_alpha + z_beta) * assumed_std / effect_size) ** 2
        
        return int(np.ceil(n))


# ============== Bayesian Price Optimization ==============

@dataclass
class BayesianPriceEstimate:
    """Bayesian price optimization result."""
    optimal_price: float
    expected_profit: float
    posterior_mean: float
    posterior_std: float
    credible_interval: Tuple[float, float]
    acquisition_value: float


class ThompsonSamplingPriceOptimizer:
    """
    Bayesian price optimization using Thompson Sampling.
    
    Balances exploration (trying new prices) and exploitation
    (using known good prices) in an optimal way.
    
    Uses a Beta-Bernoulli model for conversion rates and
    Normal-Gamma for revenue, updated in real-time.
    
    Example:
        optimizer = ThompsonSamplingPriceOptimizer(
            price_grid=np.arange(80, 120, 5)
        )
        
        for customer in customers:
            # Select price
            price = optimizer.select_price()
            
            # Observe outcome
            revenue = sell_to_customer(customer, price)
            optimizer.update(price, revenue)
    """
    
    def __init__(
        self,
        price_grid: np.ndarray,
        prior_mean: float = 0.0,
        prior_precision: float = 0.01,
        prior_a: float = 1.0,
        prior_b: float = 1.0
    ):
        self.price_grid = price_grid
        self.n_prices = len(price_grid)
        
        # Prior for mean revenue at each price (Normal)
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        
        # Posterior parameters for each price
        # Using Normal-Gamma conjugate prior
        self.mu = np.full(self.n_prices, prior_mean)  # Posterior mean
        self.kappa = np.full(self.n_prices, prior_precision)  # Precision on mean
        self.alpha = np.full(self.n_prices, prior_a)  # Shape for variance
        self.beta = np.full(self.n_prices, prior_b)  # Rate for variance
        
        # Tracking
        self.n_obs = np.zeros(self.n_prices)
        self.sum_revenue = np.zeros(self.n_prices)
        self.sum_sq_revenue = np.zeros(self.n_prices)
    
    def select_price(self) -> float:
        """
        Select price using Thompson Sampling.
        
        Sample from posterior distribution for each price,
        select price with highest sampled expected profit.
        """
        samples = np.zeros(self.n_prices)
        
        for i in range(self.n_prices):
            # Sample variance from Inverse-Gamma
            precision = np.random.gamma(self.alpha[i], 1/self.beta[i])
            variance = 1 / precision
            
            # Sample mean from Normal
            std = np.sqrt(variance / self.kappa[i])
            mean = np.random.normal(self.mu[i], std)
            
            samples[i] = mean
        
        # Select price with highest sampled profit
        best_idx = np.argmax(samples)
        return self.price_grid[best_idx]
    
    def update(self, price: float, revenue: float):
        """
        Update posterior after observing revenue.
        
        Uses Normal-Gamma Bayesian update.
        """
        # Find price index
        idx = np.argmin(np.abs(self.price_grid - price))
        
        # Update counts
        self.n_obs[idx] += 1
        n = self.n_obs[idx]
        
        # Update sufficient statistics
        self.sum_revenue[idx] += revenue
        self.sum_sq_revenue[idx] += revenue ** 2
        
        # Posterior update (Normal-Gamma)
        prior_mu = self.prior_mean
        prior_kappa = self.prior_precision
        
        sample_mean = self.sum_revenue[idx] / n
        sample_var = (self.sum_sq_revenue[idx] / n - sample_mean ** 2) if n > 1 else 0
        
        # Update posterior parameters
        self.kappa[idx] = prior_kappa + n
        self.mu[idx] = (prior_kappa * prior_mu + n * sample_mean) / self.kappa[idx]
        self.alpha[idx] = 1.0 + n / 2
        
        if n > 1:
            self.beta[idx] = 1.0 + 0.5 * (n * sample_var + 
                (prior_kappa * n / self.kappa[idx]) * (sample_mean - prior_mu)**2)
    
    def get_optimal_price(self) -> BayesianPriceEstimate:
        """
        Get current optimal price estimate.
        
        Returns the price with highest posterior mean profit.
        """
        # Posterior means
        posterior_means = self.mu.copy()
        
        # Posterior variances
        posterior_vars = self.beta / (self.alpha * self.kappa)
        posterior_stds = np.sqrt(posterior_vars)
        
        # Best price by posterior mean
        best_idx = np.argmax(posterior_means)
        
        # Credible interval (95%)
        ci_lower = posterior_means[best_idx] - 1.96 * posterior_stds[best_idx]
        ci_upper = posterior_means[best_idx] + 1.96 * posterior_stds[best_idx]
        
        # Acquisition value (expected improvement)
        current_best = np.max(posterior_means)
        z = (posterior_means - current_best) / posterior_stds
        acquisition = posterior_stds * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
        
        return BayesianPriceEstimate(
            optimal_price=self.price_grid[best_idx],
            expected_profit=posterior_means[best_idx],
            posterior_mean=posterior_means[best_idx],
            posterior_std=posterior_stds[best_idx],
            credible_interval=(ci_lower, ci_upper),
            acquisition_value=acquisition[best_idx]
        )
    
    def get_exploration_price(self) -> float:
        """
        Get price to explore (highest uncertainty).
        
        Useful for deliberate exploration phases.
        """
        posterior_vars = self.beta / (self.alpha * self.kappa)
        explore_idx = np.argmax(posterior_vars)
        return self.price_grid[explore_idx]


# ============== Causal Inference ==============

def estimate_price_effect_did(
    treatment_before: np.ndarray,
    treatment_after: np.ndarray,
    control_before: np.ndarray,
    control_after: np.ndarray,
    confidence_level: float = 0.95
) -> Dict:
    """
    Difference-in-Differences estimator for price change effects.
    
    Estimates causal effect of price change by comparing
    treated products to control products, before and after.
    
    Args:
        treatment_before: Outcomes for treated, pre-period
        treatment_after: Outcomes for treated, post-period
        control_before: Outcomes for control, pre-period
        control_after: Outcomes for control, post-period
        
    Returns:
        Dict with effect estimate and statistics
    """
    # Compute means
    mean_tb = np.mean(treatment_before)
    mean_ta = np.mean(treatment_after)
    mean_cb = np.mean(control_before)
    mean_ca = np.mean(control_after)
    
    # DiD estimate
    did_estimate = (mean_ta - mean_tb) - (mean_ca - mean_cb)
    
    # Standard error (clustered by group)
    n_t = len(treatment_before) + len(treatment_after)
    n_c = len(control_before) + len(control_after)
    
    var_t = np.var(np.concatenate([treatment_before, treatment_after]))
    var_c = np.var(np.concatenate([control_before, control_after]))
    
    se = np.sqrt(var_t / n_t + var_c / n_c)
    
    # Confidence interval
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = did_estimate - z * se
    ci_upper = did_estimate + z * se
    
    # T-test
    t_stat = did_estimate / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_t + n_c - 2))
    
    return {
        "did_estimate": did_estimate,
        "standard_error": se,
        "confidence_interval": (ci_lower, ci_upper),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < (1 - confidence_level),
        "treatment_effect_pct": did_estimate / mean_tb if mean_tb > 0 else 0
    }
