.PHONY: install data validate profile features splits reference current-sample \
        train evaluate register retrain drift monitor trigger-retrain \
        batch-predict api frontend mlflow-ui test docker-build docker-up \
        docker-down docker-logs clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

# --- Data pipeline ---
data:
	$(PYTHON) scripts/generate_synthetic_data.py

validate:
	$(PYTHON) scripts/validate_data.py

profile:
	$(PYTHON) scripts/profile_data.py

features:
	$(PYTHON) scripts/build_features.py

splits:
	$(PYTHON) scripts/create_splits.py

reference:
	$(PYTHON) scripts/create_reference_data.py

current-sample:
	$(PYTHON) scripts/create_current_sample.py

# --- Model lifecycle ---
train:
	$(PYTHON) scripts/train_model.py

evaluate:
	$(PYTHON) scripts/evaluate_model.py

register:
	$(PYTHON) scripts/register_model.py

retrain:
	$(PYTHON) scripts/retrain_model.py

trigger-retrain:
	$(PYTHON) scripts/trigger_retraining.py

batch-predict:
	$(PYTHON) scripts/batch_predict.py

# --- Monitoring ---
drift:
	$(PYTHON) scripts/run_drift_check.py

monitor:
	$(PYTHON) scripts/run_monitoring.py

# --- Local services (non-Docker) ---
api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/streamlit_app.py

mlflow-ui:
	mlflow server --backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# --- Tests ---
test:
	pytest -v

# --- Docker ---
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
