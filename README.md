# Dynamic Pricing Engine for E-Commerce

A high-performance, GPU-accelerated dynamic pricing system designed for real-time price optimization in e-commerce platforms.

## Overview

This project implements a production-ready dynamic pricing engine that leverages:

- **GPU Computing**: CUDA-accelerated price optimization and demand forecasting using PyTorch and CuPy
- **Low-Latency Infrastructure**: Sub-10ms pricing decisions with Redis caching, async processing, and optimized data pipelines
- **Advanced ML/Statistics**: Demand elasticity modeling, reinforcement learning for pricing, Bayesian optimization, and A/B testing frameworks

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DYNAMIC PRICING ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Data Lake  │───▶│  Feature     │───▶│  ML Models   │                   │
│  │  (S3/Delta)  │    │  Store       │    │  (GPU)       │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    STREAMING LAYER (Kafka)                    │           │
│  └──────────────────────────────────────────────────────────────┘           │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Price      │◀──▶│   Redis      │◀──▶│   API        │                   │
│  │   Optimizer  │    │   Cache      │    │   Gateway    │                   │
│  │   (GPU)      │    │   (< 1ms)    │    │   (FastAPI)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│                                          ┌──────────────┐                   │
│                                          │  E-Commerce  │                   │
│                                          │  Platform    │                   │
│                                          └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. GPU-Accelerated Computing (`src/gpu/`)
- **Batch Price Optimization**: Process millions of SKUs in parallel using CUDA/MPS kernels
- **Cross-Platform Support**: Automatic device detection (NVIDIA CUDA, Apple Silicon MPS, CPU fallback)
- **Demand Forecasting**: GPU-accelerated LSTM/Transformer models for time-series prediction
- **Matrix Operations**: PyTorch-based elasticity matrix computations

### 2. Machine Learning Models (`src/models/`)
- **Demand Elasticity Model**: Estimates price sensitivity across customer segments
- **Demand Forecasting**: Multi-horizon demand prediction with uncertainty quantification
- **Reinforcement Learning**: Deep Q-Network for sequential pricing decisions
- **Competitor Response Model**: Predicts competitor pricing reactions

### 3. Statistical Framework (`src/utils/`)
- **Bayesian Price Optimization**: Thompson Sampling for exploration-exploitation
- **A/B Testing Engine**: Sequential testing with early stopping
- **Causal Inference**: Price elasticity estimation with instrumental variables

### 4. Low-Latency Infrastructure (`src/api/`)
- **FastAPI Gateway**: Async endpoints with <10ms response times
- **Redis Caching**: Pre-computed prices with intelligent invalidation
- **Feature Store**: Real-time feature serving with sub-millisecond latency

## Hardware Configuration

### Tested System

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core i7-13700 (16 cores, 24 threads) |
| **RAM** | 64 GB DDR4 |
| **GPU** | NVIDIA RTX 4080 (16 GB VRAM) |
| **Storage** | 196 GB NVMe SSD |

### Supported Platforms

| Platform | Device | Status |
|----------|--------|--------|
| NVIDIA GPU | CUDA | ✅ Full support |
| Apple Silicon | MPS | ✅ Full support |
| CPU | PyTorch | ✅ Fallback mode |

The system automatically detects available hardware and selects the best device.

## Performance Targets

Benchmarked on: **Intel i7-13700 (16 cores) | 64GB RAM | NVIDIA RTX 4080 (16GB)**

| Metric | Target | Achieved |
|--------|--------|----------|
| API Response Time (p99) | <10ms | 6.1ms |
| Price Computation (1M SKUs) | <1s | 0.2s (GPU) |
| Model Inference Latency | <5ms | ~2ms |
| Cache Hit Rate | >95% | 97.2% |
| Throughput | >10k req/s | 22k req/s |
| GPU Speedup vs CPU | - | 70× |
| Peak Throughput | - | 5M products/s |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/dynamic-pricing.git
cd dynamic-pricing

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install CUDA dependencies (requires CUDA 11.8+)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x

# Start Redis
docker run -d --name redis -p 6379:6379 redis:alpine

# Run the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Quick Start

```python
from src.models.price_optimizer import DynamicPricingEngine
from src.gpu.batch_processor import GPUBatchProcessor

# Initialize the engine
engine = DynamicPricingEngine(
    use_gpu=True,
    cache_enabled=True,
    model_path="models/pricing_model.pt"
)

# Get optimal price for a single product
optimal_price = engine.get_optimal_price(
    product_id="SKU-12345",
    current_price=99.99,
    inventory_level=150,
    competitor_prices=[95.99, 102.99],
    demand_forecast=1200
)

# Batch optimization for entire catalog
processor = GPUBatchProcessor(device="auto")  # Auto-detects CUDA/MPS/CPU
optimized_prices = processor.optimize_batch(
    product_catalog,  # DataFrame with 1M+ products
    objective="revenue"  # or "profit", "market_share"
)
```

## Project Structure

```
dynamic-pricing/
├── src/
│   ├── models/
│   │   ├── demand_forecaster.py      # LSTM/Transformer demand prediction
│   │   ├── elasticity_model.py       # Price elasticity estimation
│   │   ├── price_optimizer.py        # Core optimization engine
│   │   └── reinforcement_learner.py  # DQN for dynamic pricing
│   ├── gpu/
│   │   ├── cuda_kernels.py           # Custom CUDA operations
│   │   ├── batch_processor.py        # GPU batch processing
│   │   └── tensor_ops.py             # CuPy/PyTorch utilities
│   ├── api/
│   │   ├── main.py                   # FastAPI application
│   │   ├── routes.py                 # API endpoints
│   │   └── middleware.py             # Caching, logging, auth
│   ├── data/
│   │   ├── feature_store.py          # Real-time feature serving
│   │   ├── data_pipeline.py          # ETL and streaming
│   │   └── connectors.py             # Database/Kafka connectors
│   └── utils/
│       ├── statistics.py             # Statistical tests and methods
│       ├── ab_testing.py             # Experimentation framework
│       └── monitoring.py             # Metrics and alerting
├── configs/
│   ├── model_config.yaml             # Model hyperparameters
│   └── infrastructure.yaml           # Service configuration
├── tests/
├── notebooks/
│   └── analysis.ipynb                # Exploratory analysis
├── docs/
└── requirements.txt
```

## Mathematical Framework

### Price Elasticity Model

The demand function is modeled as:

$$Q(p) = Q_0 \cdot \left(\frac{p}{p_0}\right)^{-\epsilon} \cdot e^{\beta X}$$

Where:
- $Q(p)$ = Quantity demanded at price $p$
- $\epsilon$ = Price elasticity of demand
- $X$ = Feature vector (seasonality, competitor prices, etc.)
- $\beta$ = Learned coefficients

### Optimal Price Derivation

For profit maximization with marginal cost $c$:

$$p^* = \frac{\epsilon}{\epsilon - 1} \cdot c$$

For revenue maximization with inventory constraint $I$:

$$p^* = \arg\max_p \left[ p \cdot Q(p) \right] \quad \text{s.t.} \quad Q(p) \leq I$$

### Reinforcement Learning Formulation

- **State**: $(p_t, q_t, I_t, c_t^{comp}, s_t)$ - current price, demand, inventory, competitor prices, seasonality
- **Action**: Price adjustment $\Delta p \in \{-10\%, -5\%, 0\%, +5\%, +10\%\}$
- **Reward**: $r_t = (p_t - c) \cdot q_t - \lambda \cdot \max(0, q_t - I_t)$

## Configuration

```yaml
# configs/model_config.yaml
model:
  demand_forecaster:
    architecture: "transformer"
    hidden_size: 256
    num_layers: 4
    dropout: 0.1
    sequence_length: 90
    
  elasticity:
    method: "bayesian"
    prior_mean: -1.5
    prior_std: 0.5
    
  optimizer:
    objective: "profit"
    constraints:
      min_margin: 0.15
      max_price_change: 0.20
      inventory_threshold: 50

gpu:
  device: "auto"  # Options: "auto", "cuda", "mps", "cpu"
  batch_size: 65536
  mixed_precision: true
  
cache:
  redis_url: "redis://localhost:6379"
  ttl_seconds: 300
  warmup_enabled: true
```

## Monitoring & Metrics

The system exports Prometheus metrics:

- `pricing_api_latency_seconds` - API response time histogram
- `gpu_batch_processing_time` - GPU computation duration
- `cache_hit_ratio` - Redis cache effectiveness
- `model_inference_time` - ML model latency
- `price_change_magnitude` - Distribution of price adjustments
- `revenue_impact` - Estimated revenue change from pricing decisions

## License

MIT License - see LICENSE file for details.
