"""CLI for generating the synthetic fraud-detection transaction dataset.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --config configs/synthetic_data.yaml
    python scripts/generate_synthetic_data.py --num-rows 10000 --fraud-rate 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.synthetic_data import SyntheticDataConfig, save_synthetic_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transaction data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_data.yaml",
        help="Path to a YAML config with num_rows/fraud_rate/random_seed/output_path.",
    )
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--fraud-rate", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SyntheticDataConfig:
    config_dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    if args.num_rows is not None:
        config_dict["num_rows"] = args.num_rows
    if args.fraud_rate is not None:
        config_dict["fraud_rate"] = args.fraud_rate
    if args.random_seed is not None:
        config_dict["random_seed"] = args.random_seed
    if args.output_path is not None:
        config_dict["output_path"] = args.output_path

    return SyntheticDataConfig(**config_dict)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    output_path = save_synthetic_transactions(config)
    print(
        f"Wrote {config.num_rows} rows (fraud_rate={config.fraud_rate}, "
        f"seed={config.random_seed}) to {output_path}"
    )


if __name__ == "__main__":
    main()
