# Fraud Risk Model Card 

## Model Purpose 

This model predicts the probability that a financial transaction is fraudulent or risky.

## Model Type 

Baseline model: logistic regression with preprocessing pipeline.

## Training Data 

Synthetic transaction dataset generated for portfolio demonstration.

## Target

`is_fraud`

- `0`: non-fraud
- `1`: fraud

## Features 

The model uses transaction amount, transaction hour, merchant category, customer tenure, recent transaction velocity, failed transaction rate, previous chargebacks, account age, device type, country, and external risk score.

## Evaluation Metrics

Primary metrics:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- Fraud capture rate 
- Review rate 
- Expected cost 

## Threshold

The selected decision threshold is chosen on the validation set using expected cost minimization.

## Limitations

- Dataset is synthetic.
- No real customer or transaction data is used.
- Model should not be used for real financial decisions.
- Fairness and bias checks are not fully implemented in the baseline version.

## Monitoring Plan 

Monitor:

- prediction volume
- average latency 
- review rate 
- high-risk rate 
- risk-band distribution 
- input feature drift 
- prediction drift 

## Retraining Criteria 

Retraining should be considered if:

- data drift is detected 
- prediction drift is detected 
- labeled performance degrades 
- expected cost increases 