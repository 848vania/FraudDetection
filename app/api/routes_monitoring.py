from fastapi import APIRouter, HTTPException

from app.monitoring.metrics import get_monitoring_summary

router = APIRouter(
    prefix = "/monitoring",
    tags = ['monitoring'],
)

@router.get("/summary")
def monitoring_summary() -> dict:
    """
    Return prediction monitoring summary
    """
    try:
        return get_monitoring_summary()

    except Exception as error:
        raise HTTPException(
            status_code= 500,
            detail= f"Failed to load monitoring summary: {error}",
        ) from error 