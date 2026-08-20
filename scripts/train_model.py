import argparse
import json 

from app.models.train import load_training_config, train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Train a fraud-risk prediction model."
    )

    parser.add_argument(
        "--config-path",
        default= "configs/baseline_logistic.yaml",
        help= "Path to training config YAML",
    )

    parser.add_argument(
        "--no-mlflow",
        action= "store_true",
        help= "Disable MLflow experiment tracking",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_training_config(args.config_path)

    result = train_model(
        config= config,
        use_mlflow= not args.no_mlflow,
    )

    print("Training completed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()