from typing import Any

from app.database.connection import SessionLocal, init_db
from app.database.crud import create_prediction_log


def log_prediction(
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        source: str = "api",
        latency_ms: float | None = None,
    ) -> None:
    """
    Save prediction log safety

    Logging failure should not break online prediction
    """
    try:
        init_db()

        db = SessionLocal()

        try:
            create_prediction_log(
                db= db,
                transaction= transaction,
                prediction= prediction,
                source= source,
                latency_ms= latency_ms,
            )
        finally:
            db.close()

    except Exception as error:
        print(f"Failed to log prediction: {error}")