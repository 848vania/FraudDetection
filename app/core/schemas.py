from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str | None = Field(
        default= None, 
        description= "Optimal transaction identifier.",
    )
    customer_id: str | None = Field(
        default= None,
        description= "Optional customer identifier",
    )

    transaction_amount: float = Field(..., ge=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    merchant_category: str
    customer_tenure_days: int = Field(..., ge=0)
    num_transactions_24h: int = Field(..., ge=0)
    num_failed_transactions_24h: int = Field(..., ge=0)
    avg_transaction_amount_30d: float = Field(..., ge=0)
    is_foreign_transaction: bool 
    device_type: str  
    country: str 
    previous_chargebacks: int = Field(..., ge=0)
    account_age_days: int = Field(..., ge=0)
    risk_score_external: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    transaction_id: str | None = None
    fraud_probability: float 
    risk_band: str 
    decision: str 
    threshold_used: float  
    model_name: str 
    model_version: str | None = None 
    model_alias: str | None = None 


class ModelInfoResponse(BaseModel):
    registered_model_name: str 
    model_version: str | None = None 
    alias: str | None = None 
    model_type: str | None = None 
    selected_threshold: float | None = None 
    test_roc_auc: float | None = None 
    test_pr_auc: float | None = None 
    test_precision: float | None = None 
    test_recall: float | None = None 
    test_expected_cost: float | None = None 