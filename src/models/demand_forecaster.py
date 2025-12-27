"""
Demand Forecasting Models for Dynamic Pricing

This module implements GPU-accelerated demand forecasting using:
- LSTM networks for sequential patterns
- Transformer architecture for long-range dependencies
- Quantile regression for uncertainty estimation

Models are designed for high-throughput batch inference on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ForecastConfig:
    """Configuration for demand forecasting models."""
    # Architecture
    model_type: str = "transformer"  # "lstm" or "transformer"
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Input/Output
    sequence_length: int = 90  # 90 days of history
    forecast_horizon: int = 14  # 14 days ahead
    num_features: int = 15  # price, promo, seasonality, etc.
    
    # Training
    batch_size: int = 512
    learning_rate: float = 1e-4
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerDemandForecaster(nn.Module):
    """
    Transformer-based demand forecasting model.
    
    Uses self-attention to capture complex temporal patterns and
    cross-attention for incorporating external features like promotions
    and competitor prices.
    
    Architecture:
    - Input embedding layer
    - Positional encoding
    - N Transformer encoder layers
    - Quantile regression head for uncertainty estimation
    """
    
    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.sequence_length + config.forecast_horizon)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Quantile prediction heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, config.forecast_horizon)
            )
            for _ in config.quantiles
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for demand forecasting.
        
        Args:
            x: Input features, shape (batch, seq_len, num_features)
            mask: Optional attention mask
            
        Returns:
            Dict with quantile forecasts for each horizon
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        h = self.input_projection(x)
        
        # Add positional encoding
        h = self.pos_encoder(h)
        
        # Create causal mask for autoregressive prediction
        if mask is None:
            mask = self._generate_causal_mask(seq_len, x.device)
        
        # Transformer encoding
        h = self.transformer_encoder(h, mask=mask)
        
        # Use last hidden state for prediction
        h_last = h[:, -1, :]  # (batch, hidden)
        
        # Generate quantile forecasts
        forecasts = {}
        for i, quantile in enumerate(self.config.quantiles):
            q_forecast = self.quantile_heads[i](h_last)  # (batch, horizon)
            forecasts[f"q{int(quantile*100):02d}"] = q_forecast
        
        # Point forecast is median (q50)
        forecasts["point"] = forecasts["q50"]
        
        return forecasts
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class LSTMDemandForecaster(nn.Module):
    """
    LSTM-based demand forecasting with attention mechanism.
    
    More efficient than Transformer for shorter sequences and
    smaller models, with good performance on seasonal patterns.
    """
    
    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size * 2,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Quantile heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.forecast_horizon)
            for _ in config.quantiles
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Project input
        h = self.input_projection(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(h)
        
        # Self-attention over LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last timestep and project
        h_final = self.output_projection(attn_out[:, -1, :])
        
        # Quantile forecasts
        forecasts = {}
        for i, quantile in enumerate(self.config.quantiles):
            forecasts[f"q{int(quantile*100):02d}"] = self.quantile_heads[i](h_final)
        forecasts["point"] = forecasts["q50"]
        
        return forecasts


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    
    def __init__(self, quantiles: Tuple[float, ...]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        
        for quantile in self.quantiles:
            q_key = f"q{int(quantile*100):02d}"
            pred = predictions[q_key]
            
            errors = targets - pred
            loss = torch.max(quantile * errors, (quantile - 1) * errors)
            losses.append(loss.mean())
        
        return torch.stack(losses).mean()


class DemandForecaster:
    """
    High-level demand forecasting interface.
    
    Handles model loading, inference, and post-processing.
    Optimized for low-latency batch predictions.
    
    Example:
        forecaster = DemandForecaster.load("models/demand_model.pt")
        predictions = forecaster.predict(historical_data)
    """
    
    def __init__(
        self,
        config: Optional[ForecastConfig] = None,
        device: str = "auto"
    ):
        self.config = config or ForecastConfig()
        
        # Auto-detect best device: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        if self.config.model_type == "transformer":
            self.model = TransformerDemandForecaster(self.config)
        else:
            self.model = LSTMDemandForecaster(self.config)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Enable inference optimizations (CUDA only)
        if self.device.type == "cuda":
            self.model = torch.jit.script(self.model)
            torch.cuda.empty_cache()
    
    @classmethod
    def load(cls, model_path: str, device: str = "auto") -> "DemandForecaster":
        """Load a trained model from disk."""
        checkpoint = torch.load(model_path, map_location=device)
        config = ForecastConfig(**checkpoint["config"])
        
        forecaster = cls(config, device)
        forecaster.model.load_state_dict(checkpoint["model_state_dict"])
        
        return forecaster
    
    def save(self, model_path: str):
        """Save model to disk."""
        torch.save({
            "config": self.config.__dict__,
            "model_state_dict": self.model.state_dict()
        }, model_path)
    
    @torch.no_grad()
    def predict(
        self,
        historical_features: np.ndarray,
        return_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate demand forecasts.
        
        Args:
            historical_features: Shape (n_products, seq_len, n_features)
            return_uncertainty: Whether to return quantile predictions
            
        Returns:
            Dict with 'point' forecast and optionally quantile forecasts
        """
        # Convert to tensor
        x = torch.tensor(historical_features, device=self.device, dtype=torch.float32)
        
        # Handle batching for large inputs
        batch_size = self.config.batch_size
        n_products = x.shape[0]
        
        all_forecasts = {key: [] for key in ["point", "q10", "q50", "q90"]}
        
        for i in range(0, n_products, batch_size):
            batch = x[i:i+batch_size]
            
            # Use autocast for faster inference (CUDA/MPS)
            autocast_enabled = self.device.type in ("cuda", "mps")
            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                forecasts = self.model(batch)
            
            for key in all_forecasts:
                if key in forecasts:
                    all_forecasts[key].append(forecasts[key].cpu().numpy())
        
        # Concatenate batches
        result = {
            key: np.concatenate(vals, axis=0)
            for key, vals in all_forecasts.items()
            if vals
        }
        
        if not return_uncertainty:
            result = {"point": result["point"]}
        
        return result
    
    @torch.no_grad()
    def predict_single(
        self,
        historical_features: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Predict demand for a single product (low-latency path).
        
        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound)
        """
        x = torch.tensor(
            historical_features[np.newaxis, :, :],
            device=self.device,
            dtype=torch.float32
        )
        
        forecasts = self.model(x)
        
        point = forecasts["point"][0, 0].item()  # First horizon
        lower = forecasts["q10"][0, 0].item()
        upper = forecasts["q90"][0, 0].item()
        
        return point, lower, upper


class FeatureEngineer:
    """
    Feature engineering for demand forecasting.
    
    Generates features from raw historical data:
    - Lag features
    - Rolling statistics
    - Seasonality indicators
    - Price features
    - Promotional features
    """
    
    def __init__(self, config: ForecastConfig):
        self.config = config
    
    def create_features(
        self,
        price_history: np.ndarray,
        quantity_history: np.ndarray,
        promo_history: np.ndarray,
        date_index: np.ndarray
    ) -> np.ndarray:
        """
        Create feature matrix for forecasting.
        
        Args:
            price_history: Historical prices (n_products, n_days)
            quantity_history: Historical quantities (n_products, n_days)
            promo_history: Promotion flags (n_products, n_days)
            date_index: Date array for seasonality
            
        Returns:
            Feature matrix (n_products, seq_len, n_features)
        """
        n_products, n_days = price_history.shape
        seq_len = self.config.sequence_length
        
        # Initialize feature array
        features = []
        
        # 1. Price features
        features.append(price_history[:, -seq_len:, np.newaxis])
        
        # Price change
        price_change = np.diff(price_history, axis=1, prepend=price_history[:, :1])
        features.append(price_change[:, -seq_len:, np.newaxis])
        
        # 2. Quantity features (log-transformed)
        log_quantity = np.log1p(quantity_history)
        features.append(log_quantity[:, -seq_len:, np.newaxis])
        
        # Quantity lag features
        for lag in [1, 7, 14, 28]:
            lagged = np.roll(log_quantity, lag, axis=1)
            features.append(lagged[:, -seq_len:, np.newaxis])
        
        # 3. Rolling statistics
        for window in [7, 14, 28]:
            rolling_mean = self._rolling_mean(log_quantity, window)
            rolling_std = self._rolling_std(log_quantity, window)
            features.append(rolling_mean[:, -seq_len:, np.newaxis])
            features.append(rolling_std[:, -seq_len:, np.newaxis])
        
        # 4. Promotion features
        features.append(promo_history[:, -seq_len:, np.newaxis])
        
        # 5. Seasonality features from dates
        day_of_week = self._day_of_week_encoding(date_index[-seq_len:])
        month_encoding = self._month_encoding(date_index[-seq_len:])
        
        # Broadcast to all products
        features.append(np.tile(day_of_week[np.newaxis, :, :], (n_products, 1, 1)))
        features.append(np.tile(month_encoding[np.newaxis, :, :], (n_products, 1, 1)))
        
        # Concatenate all features
        return np.concatenate(features, axis=2).astype(np.float32)
    
    def _rolling_mean(self, x: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean along axis 1."""
        result = np.zeros_like(x)
        for i in range(x.shape[1]):
            start = max(0, i - window + 1)
            result[:, i] = x[:, start:i+1].mean(axis=1)
        return result
    
    def _rolling_std(self, x: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation along axis 1."""
        result = np.zeros_like(x)
        for i in range(x.shape[1]):
            start = max(0, i - window + 1)
            result[:, i] = x[:, start:i+1].std(axis=1)
        return result
    
    def _day_of_week_encoding(self, dates: np.ndarray) -> np.ndarray:
        """One-hot encode day of week."""
        # Assuming dates are integers representing days
        dow = dates % 7
        encoding = np.zeros((len(dates), 7))
        encoding[np.arange(len(dates)), dow] = 1
        return encoding
    
    def _month_encoding(self, dates: np.ndarray) -> np.ndarray:
        """Sinusoidal month encoding."""
        # Assuming dates can be converted to month (0-11)
        month = (dates // 30) % 12
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)
        return np.stack([sin_month, cos_month], axis=1)
