import json 
from pathlib import Path
from typing import Any

import pandas as pd 
import requests


API_BASE_URL = 'http://localhost:8000'
RESULTS_DIR = Path('data/results')


def post_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Call FastAPI prediction endpoint
    """
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json= payload,
        timeout= 60,
    )
    response.raise_for_status()

    return response.json()


def get_model_info() -> dict[str, Any]:
    """
    Call model info endpoint
    """
    response = requests.get(
        f"{API_BASE_URL}/model/info",
        timeout= 30,
    )
    response.raise_for_status()

    return response.json()


def get_monitoring_summary() -> dict[str, Any]:
    """
    Call monitoring summary endpoint
    """
    response = requests.get(
        f"{API_BASE_URL}/monitoring/summary",
        timeout=30,
    )
    response.raise_for_status()

    return response.json()


def load_json_file(path: str | Path) -> dict[str, Any] | list | None:
    """
    Load JSON file if it exists
    """
    path = Path(path)

    if not path.exists():
        return None 

    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def load_evaluation_metrics() -> dict[str, Any]:
    return load_json_file(RESULTS_DIR / 'evaluation_metrics.json') or {}


def load_threshold_report() -> dict[str, Any]:
    return load_json_file(RESULTS_DIR / 'threshold_report.json') or {}


def load_drift_summary() -> dict[str, Any]:
    return load_json_file(RESULTS_DIR / 'drift_summary.json') or {}


def load_retraining_report() -> dict[str, Any]:
    return load_json_file(RESULTS_DIR / 'retraining_report.json') or {}


def load_retraining_trigger_report() -> dict[str, Any]:
    return load_json_file(RESULTS_DIR / 'retraining_trigger_report.json') or {}


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "-"

    return f"{float(value) * 100:.1f}%"


def format_cost(value: float | int | None) -> str:
    if value is None:
        return "-"

    return f"{float(value):.2f}"


def format_latency_ms(value: float | int | None) -> str:
    if value is None: 
        return "-"

    return  f"{float(value):.1f} ms"


def dict_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert simple dictionary to dataframe with key/value columns
    """
    return pd.DataFrame(
        [{'metric': key, 'value': value} for key, value in data.items()]
    )