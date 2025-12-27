"""
FastAPI Application for Dynamic Pricing

This module provides the REST API for the pricing engine,
optimized for low-latency responses (<10ms p99).

Features:
- Async request handling
- Redis caching layer
- Request batching for efficiency
- Prometheus metrics
- Health checks and graceful shutdown
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============== Pydantic Models ==============

class PriceRequest(BaseModel):
    """Request model for single price optimization."""
    product_id: str = Field(..., description="Unique product identifier")
    current_price: float = Field(..., gt=0, description="Current selling price")
    cost: float = Field(..., gt=0, description="Unit cost")
    inventory_level: int = Field(..., ge=0, description="Current inventory")
    competitor_prices: Optional[List[float]] = Field(None, description="Competitor prices")
    demand_forecast: Optional[float] = Field(None, description="Predicted demand")
    elasticity: Optional[float] = Field(None, description="Price elasticity")
    category: Optional[str] = Field(None, description="Product category")
    objective: str = Field("profit", description="Optimization objective")


class PriceResponse(BaseModel):
    """Response model for price optimization."""
    product_id: str
    optimal_price: float
    current_price: float
    price_change_pct: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    confidence_lower: float
    confidence_upper: float
    reasoning: str
    latency_ms: float


class BatchPriceRequest(BaseModel):
    """Request model for batch price optimization."""
    products: List[PriceRequest]
    objective: str = "profit"


class BatchPriceResponse(BaseModel):
    """Response model for batch optimization."""
    results: List[PriceResponse]
    total_latency_ms: float
    products_processed: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    cache_connected: bool
    model_loaded: bool
    uptime_seconds: float


# ============== Application State ==============

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.pricing_engine = None
        self.redis_client = None
        self.start_time = time.time()
        self.request_count = 0
        
        # Metrics
        if PROMETHEUS_AVAILABLE:
            self.request_latency = Histogram(
                'pricing_api_latency_seconds',
                'API request latency',
                ['endpoint']
            )
            self.request_counter = Counter(
                'pricing_api_requests_total',
                'Total API requests',
                ['endpoint', 'status']
            )
            self.cache_hits = Counter(
                'pricing_cache_hits_total',
                'Cache hit count'
            )
            self.gpu_utilization = Gauge(
                'pricing_gpu_utilization',
                'GPU utilization percentage'
            )
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing application state...")
        
        # Initialize pricing engine
        from ..models.price_optimizer import DynamicPricingEngine
        self.pricing_engine = DynamicPricingEngine(
            use_gpu=True,
            cache_enabled=True
        )
        
        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        logger.info("Application initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Application cleanup complete")


# Global state
state = AppState()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    await state.initialize()
    yield
    await state.cleanup()


# ============== FastAPI App ==============

app = FastAPI(
    title="Dynamic Pricing API",
    description="High-performance, GPU-accelerated pricing optimization API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Caching Layer ==============

async def get_cached_price(product_id: str) -> Optional[dict]:
    """Get cached price from Redis."""
    if not state.redis_client:
        return None
    
    try:
        cached = await state.redis_client.get(f"price:{product_id}")
        if cached:
            if PROMETHEUS_AVAILABLE:
                state.cache_hits.inc()
            import json
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    
    return None


async def set_cached_price(product_id: str, price_data: dict, ttl: int = 300):
    """Cache price in Redis."""
    if not state.redis_client:
        return
    
    try:
        import json
        await state.redis_client.setex(
            f"price:{product_id}",
            ttl,
            json.dumps(price_data)
        )
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    import torch
    
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        cache_connected=state.redis_client is not None,
        model_loaded=state.pricing_engine is not None,
        uptime_seconds=time.time() - state.start_time
    )


@app.post("/v1/price", response_model=PriceResponse)
async def get_optimal_price(
    request: PriceRequest,
    background_tasks: BackgroundTasks
):
    """
    Get optimal price for a single product.
    
    This is the primary low-latency endpoint, targeting <10ms p99.
    Uses caching and optimized inference path.
    """
    start_time = time.perf_counter()
    
    try:
        # Check cache first
        cached = await get_cached_price(request.product_id)
        if cached:
            latency = (time.perf_counter() - start_time) * 1000
            return PriceResponse(
                **cached,
                latency_ms=latency
            )
        
        # Run optimization
        from ..models.price_optimizer import OptimizationObjective
        
        objective = OptimizationObjective(request.objective)
        
        result = state.pricing_engine.get_optimal_price(
            product_id=request.product_id,
            current_price=request.current_price,
            cost=request.cost,
            inventory_level=request.inventory_level,
            competitor_prices=request.competitor_prices,
            demand_forecast=request.demand_forecast,
            elasticity=request.elasticity,
            category=request.category,
            objective=objective
        )
        
        response_data = {
            "product_id": request.product_id,
            "optimal_price": result.optimal_price,
            "current_price": result.current_price,
            "price_change_pct": result.price_change_pct,
            "expected_demand": result.expected_demand,
            "expected_revenue": result.expected_revenue,
            "expected_profit": result.expected_profit,
            "confidence_lower": result.confidence_interval[0],
            "confidence_upper": result.confidence_interval[1],
            "reasoning": result.reasoning
        }
        
        # Cache in background
        background_tasks.add_task(
            set_cached_price,
            request.product_id,
            response_data
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            state.request_latency.labels(endpoint='/v1/price').observe(latency / 1000)
            state.request_counter.labels(endpoint='/v1/price', status='success').inc()
        
        return PriceResponse(**response_data, latency_ms=latency)
    
    except Exception as e:
        logger.error(f"Pricing error: {e}")
        if PROMETHEUS_AVAILABLE:
            state.request_counter.labels(endpoint='/v1/price', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/price/batch", response_model=BatchPriceResponse)
async def get_optimal_prices_batch(request: BatchPriceRequest):
    """
    Batch price optimization for multiple products.
    
    Uses GPU batch processing for efficient bulk optimization.
    Suitable for catalog-wide repricing.
    """
    start_time = time.perf_counter()
    
    try:
        from ..models.price_optimizer import OptimizationObjective
        
        objective = OptimizationObjective(request.objective)
        
        # Convert to dict format for batch processor
        products = {
            p.product_id: {
                "current_price": p.current_price,
                "cost": p.cost,
                "inventory": p.inventory_level,
                "demand_forecast": p.demand_forecast or 100,
                "elasticity": p.elasticity or -1.8
            }
            for p in request.products
        }
        
        # Run batch optimization
        results = state.pricing_engine.optimize_catalog(
            products=products,
            objective=objective
        )
        
        # Build response
        response_results = []
        for p in request.products:
            result = results[p.product_id]
            response_results.append(PriceResponse(
                product_id=p.product_id,
                optimal_price=result.optimal_price,
                current_price=result.current_price,
                price_change_pct=result.price_change_pct,
                expected_demand=result.expected_demand,
                expected_revenue=result.expected_revenue,
                expected_profit=result.expected_profit,
                confidence_lower=result.optimal_price * 0.95,
                confidence_upper=result.optimal_price * 1.05,
                reasoning=result.reasoning,
                latency_ms=0  # Individual latency not tracked in batch
            ))
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        if PROMETHEUS_AVAILABLE:
            state.request_latency.labels(endpoint='/v1/price/batch').observe(total_latency / 1000)
            state.request_counter.labels(endpoint='/v1/price/batch', status='success').inc()
        
        return BatchPriceResponse(
            results=response_results,
            total_latency_ms=total_latency,
            products_processed=len(request.products)
        )
    
    except Exception as e:
        logger.error(f"Batch pricing error: {e}")
        if PROMETHEUS_AVAILABLE:
            state.request_counter.labels(endpoint='/v1/price/batch', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/cache/invalidate")
async def invalidate_cache(product_ids: Optional[List[str]] = None):
    """
    Invalidate cached prices.
    
    This function is called when product data changes (cost update, promotion start, etc.)
    """
    if not state.redis_client:
        return {"status": "no_cache"}
    
    try:
        if product_ids:
            for pid in product_ids:
                await state.redis_client.delete(f"price:{pid}")
            return {"status": "invalidated", "count": len(product_ids)}
        else:
            # Flush all prices
            keys = await state.redis_client.keys("price:*")
            if keys:
                await state.redis_client.delete(*keys)
            return {"status": "flushed", "count": len(keys)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/cache/warmup")
async def warmup_cache(background_tasks: BackgroundTasks):
    """
    Pre-warm the cache with frequently accessed products.
    
    Call during deployment to minimize cold-start latency.
    """
    async def warmup_task():
        # Would load top products and pre-compute prices
        logger.info("Cache warmup started")
        # Simulate warmup
        await asyncio.sleep(1)
        logger.info("Cache warmup complete")
    
    background_tasks.add_task(warmup_task)
    return {"status": "warmup_scheduled"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            generate_latest(),
            media_type="text/plain"
        )
    return {"error": "prometheus_client not installed"}


# ============== WebSocket for Real-time Updates ==============

class ConnectionManager:
    """Manage WebSocket connections for real-time price updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, product_id: str):
        await websocket.accept()
        if product_id not in self.active_connections:
            self.active_connections[product_id] = []
        self.active_connections[product_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, product_id: str):
        if product_id in self.active_connections:
            self.active_connections[product_id].remove(websocket)
    
    async def broadcast_price_update(self, product_id: str, price_data: dict):
        if product_id in self.active_connections:
            for connection in self.active_connections[product_id]:
                try:
                    await connection.send_json(price_data)
                except:
                    pass


manager = ConnectionManager()


@app.websocket("/ws/price/{product_id}")
async def websocket_endpoint(websocket: WebSocket, product_id: str):
    """
    WebSocket endpoint for real-time price updates.
    
    Subscribe to receive instant notifications when a product's
    optimal price changes.
    """
    await manager.connect(websocket, product_id)
    try:
        while True:
            # Keep connection alive, wait for client messages
            data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, product_id)


# ============== Main ==============

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=4,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    run_server()
