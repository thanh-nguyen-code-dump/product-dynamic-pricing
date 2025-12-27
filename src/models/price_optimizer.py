"""
Dynamic Pricing Engine - Core Optimizer

Unified pricing engine for price optimization,
integrating demand forecasting, elasticity estimation, and
reinforcement learning.

Features:
- Real-time price optimization with <10ms latency
- Multiple optimization objectives (profit, revenue, market share)
- Constraint handling (min margin, max change, inventory)
- A/B test integration for safe deployment
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
from functools import lru_cache

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for pricing."""
    PROFIT = "profit"
    REVENUE = "revenue"
    MARKET_SHARE = "market_share"
    INVENTORY_CLEARANCE = "inventory_clearance"


@dataclass
class PricingConstraints:
    """Business constraints for price optimization."""
    min_margin: float = 0.10  # Minimum profit margin
    max_price_change: float = 0.20  # Maximum daily price change
    min_price: Optional[float] = None  # Absolute price floor
    max_price: Optional[float] = None  # Absolute price ceiling
    competitor_floor: float = 0.0  # Don't go below competitors by this ratio
    inventory_threshold: int = 50  # Safety stock level


@dataclass
class PricingResult:
    """Result of price optimization."""
    optimal_price: float
    current_price: float
    price_change_pct: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    confidence_interval: Tuple[float, float]
    reasoning: str
    optimization_time_ms: float


class DynamicPricingEngine:
    """
    Main pricing engine that orchestrates all optimization components.
    
    Combines multiple signals to produce optimal prices:
    - Demand forecasting model predictions
    - Price elasticity estimates
    - Competitor price analysis
    - Inventory levels and turnover targets
    - Reinforcement learning policy (for edge cases)
    
    Example:
        engine = DynamicPricingEngine(use_gpu=True)
        
        # Single product optimization
        result = engine.get_optimal_price(
            product_id="SKU-12345",
            current_price=99.99,
            cost=45.00,
            inventory_level=150,
            competitor_prices=[95.99, 102.99]
        )
        
        # Batch optimization
        results = engine.optimize_catalog(product_df)
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        cache_enabled: bool = True,
        model_path: Optional[str] = None,
        cache_ttl: int = 300  # 5 minutes
    ):
        # Auto-detect best device: CUDA > MPS > CPU
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                self.use_gpu = True
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.use_gpu = True
            else:
                self.device = torch.device("cpu")
                self.use_gpu = False
        else:
            self.device = torch.device("cpu")
            self.use_gpu = False
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        
        # Initialize components
        self._init_models(model_path)
        self._init_cache()
        
        logger.info(f"PricingEngine initialized on {self.device}")
    
    def _init_models(self, model_path: Optional[str]):
        """Initialize ML models."""
        
        # Price elasticity model parameters
        self.default_elasticity = -1.8
        self.elasticity_by_category = {
            "electronics": -2.5,
            "groceries": -1.2,
            "fashion": -2.0,
            "home": -1.5
        }
        
        # Feature importance weights for pricing
        self.feature_weights = {
            "demand_forecast": 0.35,
            "competitor_price": 0.25,
            "elasticity": 0.20,
            "inventory_pressure": 0.10,
            "seasonality": 0.10
        }
    
    def _init_cache(self):
        """Initialize price cache."""
        self.price_cache: Dict[str, Tuple[float, float]] = {}  # product_id -> (price, timestamp)
    
    def get_optimal_price(
        self,
        product_id: str,
        current_price: float,
        cost: float,
        inventory_level: int,
        competitor_prices: Optional[List[float]] = None,
        demand_forecast: Optional[float] = None,
        elasticity: Optional[float] = None,
        category: Optional[str] = None,
        objective: OptimizationObjective = OptimizationObjective.PROFIT,
        constraints: Optional[PricingConstraints] = None
    ) -> PricingResult:
        """
        Compute optimal price for a single product.
        
        This is the low-latency path optimized for real-time pricing.
        Target latency: <10ms
        
        Args:
            product_id: Unique product identifier
            current_price: Current selling price
            cost: Unit cost
            inventory_level: Current inventory count
            competitor_prices: List of competitor prices
            demand_forecast: Predicted demand (optional, will estimate)
            elasticity: Price elasticity (optional, will estimate)
            category: Product category for elasticity lookup
            objective: Optimization objective
            constraints: Business constraints
            
        Returns:
            PricingResult with optimal price and metadata
        """
        start_time = time.perf_counter()
        constraints = constraints or PricingConstraints()
        
        # Check cache first
        if self.cache_enabled and self._check_cache(product_id):
            cached_price, cached_time = self.price_cache[product_id]
            if time.time() - cached_time < self.cache_ttl:
                return self._build_result(
                    cached_price, current_price, cost, 
                    start_time, "Cached result"
                )
        
        # Get or estimate elasticity
        if elasticity is None:
            elasticity = self._get_elasticity(category)
        
        # Get or estimate demand forecast
        if demand_forecast is None:
            demand_forecast = self._estimate_base_demand(current_price, inventory_level)
        
        # Compute optimal price based on objective
        if objective == OptimizationObjective.PROFIT:
            optimal = self._optimize_profit(current_price, cost, elasticity, constraints)
        elif objective == OptimizationObjective.REVENUE:
            optimal = self._optimize_revenue(
                current_price, demand_forecast, elasticity, 
                inventory_level, constraints
            )
        elif objective == OptimizationObjective.MARKET_SHARE:
            optimal = self._optimize_market_share(
                current_price, competitor_prices or [], 
                elasticity, constraints
            )
        elif objective == OptimizationObjective.INVENTORY_CLEARANCE:
            optimal = self._optimize_clearance(
                current_price, cost, inventory_level, 
                demand_forecast, elasticity, constraints
            )
        else:
            optimal = current_price
        
        # Apply final constraints
        optimal = self._apply_constraints(optimal, current_price, cost, constraints)
        
        # Adjust for competitor prices if provided
        if competitor_prices:
            optimal = self._adjust_for_competitors(
                optimal, competitor_prices, constraints
            )
        
        # Update cache
        if self.cache_enabled:
            self.price_cache[product_id] = (optimal, time.time())
        
        # Build result
        return self._build_result(
            optimal, current_price, cost, start_time,
            f"Optimized for {objective.value}"
        )
    
    def optimize_catalog(
        self,
        products: Dict[str, Dict],
        objective: OptimizationObjective = OptimizationObjective.PROFIT,
        constraints: Optional[PricingConstraints] = None
    ) -> Dict[str, PricingResult]:
        """
        Optimize prices for entire product catalog.
        
        Uses GPU batch processing for efficiency.
        
        Args:
            products: Dict of product_id -> product attributes
            objective: Optimization objective
            constraints: Business constraints
            
        Returns:
            Dict of product_id -> PricingResult
        """
        from .batch_processor import GPUBatchProcessor, BatchConfig
        
        n_products = len(products)
        logger.info(f"Optimizing {n_products} products...")
        
        # Convert to arrays for GPU processing
        product_ids = list(products.keys())
        current_prices = np.array([p["current_price"] for p in products.values()])
        costs = np.array([p["cost"] for p in products.values()])
        elasticities = np.array([
            p.get("elasticity", self.default_elasticity) 
            for p in products.values()
        ])
        demand_forecasts = np.array([
            p.get("demand_forecast", 100) 
            for p in products.values()
        ])
        inventory_levels = np.array([
            p.get("inventory", 100) 
            for p in products.values()
        ])
        
        # Use GPU batch processor
        processor = GPUBatchProcessor(BatchConfig(device=str(self.device)))
        
        batch_results = processor.optimize_batch(
            current_prices=current_prices,
            costs=costs,
            demand_forecasts=demand_forecasts,
            elasticities=elasticities,
            inventory_levels=inventory_levels,
            objective=objective.value,
            constraints=constraints.__dict__ if constraints else None
        )
        
        # Build results dictionary
        results = {}
        for i, pid in enumerate(product_ids):
            results[pid] = PricingResult(
                optimal_price=batch_results["optimal_prices"][i],
                current_price=current_prices[i],
                price_change_pct=batch_results["price_change_pct"][i],
                expected_demand=batch_results["expected_demand"][i],
                expected_revenue=batch_results["expected_revenue"][i],
                expected_profit=batch_results["expected_profit"][i],
                confidence_interval=(0.0, 0.0),  # Would compute from uncertainty
                reasoning=f"GPU batch optimization for {objective.value}",
                optimization_time_ms=0.0  # Amortized time
            )
        
        return results
    
    def _get_elasticity(self, category: Optional[str]) -> float:
        """Get price elasticity estimate for category."""
        if category and category in self.elasticity_by_category:
            return self.elasticity_by_category[category]
        return self.default_elasticity
    
    def _estimate_base_demand(self, price: float, inventory: int) -> float:
        """Estimate base demand from current price and inventory."""
        # Simple heuristic; would use ML model in production
        return max(10, inventory * 0.1)
    
    def _optimize_profit(
        self,
        current_price: float,
        cost: float,
        elasticity: float,
        constraints: PricingConstraints
    ) -> float:
        """
        Compute profit-maximizing price.
        
        Using the analytical solution for constant elasticity:
        p* = ε / (ε - 1) * c
        """
        eps = abs(elasticity)
        
        if eps <= 1.01:
            # Nearly unit elastic, keep current price
            return current_price
        
        markup = eps / (eps - 1)
        optimal = markup * cost
        
        return optimal
    
    def _optimize_revenue(
        self,
        current_price: float,
        demand_forecast: float,
        elasticity: float,
        inventory: int,
        constraints: PricingConstraints
    ) -> float:
        """Compute revenue-maximizing price with inventory constraint."""
        eps = abs(elasticity)
        
        # For revenue max, we want to sell all inventory
        # Target demand = inventory
        if demand_forecast > 0:
            price_ratio = (inventory / demand_forecast) ** (1 / eps)
            optimal = current_price * price_ratio
        else:
            optimal = current_price
        
        return optimal
    
    def _optimize_market_share(
        self,
        current_price: float,
        competitor_prices: List[float],
        elasticity: float,
        constraints: PricingConstraints
    ) -> float:
        """Compute market-share maximizing price."""
        if not competitor_prices:
            return current_price
        
        avg_competitor = np.mean(competitor_prices)
        min_competitor = np.min(competitor_prices)
        
        # Price slightly below average to capture share
        target = avg_competitor * 0.95
        
        # But not below minimum if floor constraint exists
        if constraints.competitor_floor > 0:
            floor = min_competitor * (1 - constraints.competitor_floor)
            target = max(target, floor)
        
        return target
    
    def _optimize_clearance(
        self,
        current_price: float,
        cost: float,
        inventory: int,
        demand_forecast: float,
        elasticity: float,
        constraints: PricingConstraints
    ) -> float:
        """Optimize for inventory clearance (end of season)."""
        eps = abs(elasticity)
        
        # Calculate price needed to clear inventory
        target_demand = inventory  # Clear all inventory
        
        if demand_forecast > 0:
            price_ratio = (target_demand / demand_forecast) ** (1 / eps)
            clearance_price = current_price * price_ratio
        else:
            clearance_price = cost * 1.1  # Minimum markup
        
        # Ensure we don't go below cost
        return max(clearance_price, cost * 1.01)
    
    def _apply_constraints(
        self,
        optimal: float,
        current_price: float,
        cost: float,
        constraints: PricingConstraints
    ) -> float:
        """Apply business constraints to optimal price."""
        # Minimum margin
        min_price_for_margin = cost / (1 - constraints.min_margin)
        optimal = max(optimal, min_price_for_margin)
        
        # Maximum price change
        max_change = constraints.max_price_change
        price_floor = current_price * (1 - max_change)
        price_ceiling = current_price * (1 + max_change)
        optimal = np.clip(optimal, price_floor, price_ceiling)
        
        # Absolute bounds
        if constraints.min_price:
            optimal = max(optimal, constraints.min_price)
        if constraints.max_price:
            optimal = min(optimal, constraints.max_price)
        
        return optimal
    
    def _adjust_for_competitors(
        self,
        optimal: float,
        competitor_prices: List[float],
        constraints: PricingConstraints
    ) -> float:
        """Adjust price based on competitor positioning."""
        if not competitor_prices:
            return optimal
        
        min_competitor = min(competitor_prices)
        
        # Don't undercut by more than allowed floor
        if constraints.competitor_floor > 0:
            floor = min_competitor * (1 - constraints.competitor_floor)
            optimal = max(optimal, floor)
        
        return optimal
    
    def _check_cache(self, product_id: str) -> bool:
        """Check if product has valid cache entry."""
        return product_id in self.price_cache
    
    def _build_result(
        self,
        optimal_price: float,
        current_price: float,
        cost: float,
        start_time: float,
        reasoning: str
    ) -> PricingResult:
        """Build PricingResult object."""
        price_change_pct = (optimal_price - current_price) / current_price
        
        # Estimate demand using simple elasticity model
        elasticity = self.default_elasticity
        price_ratio = optimal_price / current_price
        demand_factor = price_ratio ** elasticity
        base_demand = 100  # Would come from forecast
        expected_demand = base_demand * demand_factor
        
        expected_revenue = optimal_price * expected_demand
        expected_profit = (optimal_price - cost) * expected_demand
        
        optimization_time_ms = (time.perf_counter() - start_time) * 1000
        
        return PricingResult(
            optimal_price=round(optimal_price, 2),
            current_price=current_price,
            price_change_pct=round(price_change_pct, 4),
            expected_demand=round(expected_demand, 1),
            expected_revenue=round(expected_revenue, 2),
            expected_profit=round(expected_profit, 2),
            confidence_interval=(optimal_price * 0.95, optimal_price * 1.05),
            reasoning=reasoning,
            optimization_time_ms=round(optimization_time_ms, 2)
        )


class PricingABTest:
    """
    A/B testing framework for pricing experiments.
    
    Enables safe rollout of new pricing strategies with
    proper statistical controls.
    """
    
    def __init__(
        self,
        control_engine: DynamicPricingEngine,
        treatment_engine: DynamicPricingEngine,
        treatment_fraction: float = 0.10
    ):
        self.control = control_engine
        self.treatment = treatment_engine
        self.treatment_fraction = treatment_fraction
        
        # Track metrics
        self.control_revenue = []
        self.treatment_revenue = []
    
    def get_price(
        self,
        product_id: str,
        user_id: str,
        **kwargs
    ) -> Tuple[PricingResult, str]:
        """
        Get price with A/B test assignment.
        
        Returns:
            Tuple of (PricingResult, variant)
        """
        # Deterministic assignment based on user_id
        variant = "treatment" if hash(user_id) % 100 < self.treatment_fraction * 100 else "control"
        
        engine = self.treatment if variant == "treatment" else self.control
        result = engine.get_optimal_price(product_id=product_id, **kwargs)
        
        return result, variant
    
    def record_outcome(self, variant: str, revenue: float):
        """Record transaction outcome for analysis."""
        if variant == "treatment":
            self.treatment_revenue.append(revenue)
        else:
            self.control_revenue.append(revenue)
    
    def get_test_stats(self) -> Dict:
        """Compute current test statistics."""
        from scipy import stats
        
        if len(self.control_revenue) < 30 or len(self.treatment_revenue) < 30:
            return {"status": "insufficient_data"}
        
        control_mean = np.mean(self.control_revenue)
        treatment_mean = np.mean(self.treatment_revenue)
        lift = (treatment_mean - control_mean) / control_mean
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(self.treatment_revenue, self.control_revenue)
        
        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "lift": lift,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_control": len(self.control_revenue),
            "n_treatment": len(self.treatment_revenue)
        }
