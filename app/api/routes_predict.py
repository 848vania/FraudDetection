import time 

from fastapi import APIRouter, HTTPException

from app.core.schemas import PredictionResponse, TransactionInput
from app.models.predict import predict_transaction


router = APIRouter(
    prefix= "",
    tags = ['prediction'],
)


def _latency_ms(start_time: float) -> float:
    """
    Calculate elapsed time in milliseconds
    """
    return round((time.perf_counter() - start_time)*1000, 2)


@router.post(
    "/predict",
    response_model= PredictionResponse,
)
def predict(transaction: TransactionInput) -> dict:
    """
    Predict fraud risk for a single transaction
    """
    start_time = time.perf_counter()

    try:
        transaction_dict = transaction.model_dump()

        prediction = predict_transaction(transaction_dict)

        prediction['latency_ms'] = _latency_ms(start_time)

        return prediction

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail= f"Prediction failed: {error}",
        ) from error 


