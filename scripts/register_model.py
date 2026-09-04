import argparse
import json

from app.models.registry import register_model_version
from app.models.train import load_training_config


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Register trained fraud-risk model"
    )

    parser.add_argument(
        "--config-path",
        default= "configs/baseline_logistic.yaml",
        help= "Path to model config YAML",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_training_config(args.config_path)
    metadata = register_model_version(config)

    print("Model registered.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()