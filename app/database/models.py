from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.database.connection import Base 


class PredictionLog(Base):
    __tablename__  = "prediction_logs"

    id = Column(Integer, primary_key=True, index= True)

    transaction_id = Column(String, nullable=True, index= True)
    customer_id = Column(String, nullable=True, index= True)

    input_json = Column(Text, nullable=False)
    prediction_json = Column(Text, nullable=False)

    fraud_probability = Column(Float, nullable=False)
    risk_band = Column(String, nullable= False)
    decision = Column(String, nullable= False)
    threshold_used = Column(Float, nullable= False)

    model_name = Column(String, nullable= False)
    model_version = Column(String, nullable= True)
    model_alias = Column(String, nullable=True)

    latency_ms = Column(Float, nullable=True)
    source = Column(String, nullable=False, default='api')

    created_at = Column(
        DateTime,
        nullable=False,
        default= lambda: datetime.now(timezone.utc)
    )