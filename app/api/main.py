from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_model import router as model_router
from app.api.routes_monitoring import router as monitoring_router
from app.api.routes_predict import router as predict_router
from app.core.schemas import HealthResponse
from app.models.predict import get_cached_model, get_cached_model_metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up model and metadata cache on startup
    """
    get_cached_model()
    get_cached_model_metadata()
    yield


app = FastAPI(
    title = "Fraud Risk Prediction API",
    description= (
        "API for scoring transaction fraud risk using a registered "
        "machine learning model."
    ),
    version= "0.1.0",
    lifespan= lifespan,
)


@app.get(
    "/health",
    response_model= HealthResponse,
)
def health() -> dict:
    """
    Health check endpoint
    """
    return {
        'status': 'ok',
        'service': 'fraud-risk-api',
    }


app.include_router(predict_router)
app.include_router(model_router)
app.include_router(monitoring_router)