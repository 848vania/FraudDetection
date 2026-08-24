from fastapi import APIRouter, HTTPException

from app.core.schemas import ModelInfoResponse
from app.models.predict import get_model_info


router = APIRouter(
    prefix= '/model',
    tags= ['model'],
)


@router.get(
    "/info",
    response_model= ModelInfoResponse,
)
def model_info() -> dict:
    """
    Return current model metadata
    """
    try:
        return get_model_info()

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail= f"Failed to load model info: {error}",
        ) from error