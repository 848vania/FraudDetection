import argparse
import json 

from app.models.evaluate import evaluate_model
from app.models.train import load_training_config


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Evaluate fraud-risk model and tune threshold"
    )

    parser.add_argument(
        "--config-path",
        default= 'configs/baseline_logistic.yaml',
        help= 'Path to model config YAML',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_training_config(args.config_path)

    report = evaluate_model(config)

    print("Evaluation completed")
    print(json.dumps(report,  indent=2))


if __name__ == "__main__":
    main()