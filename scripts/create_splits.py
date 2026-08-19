import argparse

from app.data.ingest import load_transactions
from app.data.splitting import create_and_save_splits
from app.features.feature_schema import TARGET_COLUMN


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Create train/validation/test splits"
    )

    parser.add_argument(
        "--input-path",
        default= "data/processed/features.csv",
        help= "Path to feature CSV",
    )

    parser.add_argument(
        "--output-path",
        default= "data/processed/",
        help = "Directory where train/valid/test CSV files will be saved"
    )

    parser.add_argument(
        "--target-col",
        default= TARGET_COLUMN,
        help= "Target column name"
    )

    parser.add_argument(
        "--test-size",
        type= float,
        default= 0.2,
        help = "Fraction of full dataset used for test split"
    )

    parser.add_argument(
        "--valid-size",
        type= float,
        default= 0.2,
        help = "Fraction of full dataset used for validation split"
    )

    parser.add_argument(
        "--random-state",
        type= int,
        default= 42,
        help = "Random seed for reproducible splits"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_transactions(args.input_path)

    summary = create_and_save_splits(
        df = df,
        output_dir= args.output_path,
        target_col= args.target_col,
        test_size = args.test_size,
        valid_size= args.valid_size,
        random_state= args.random_state,
    )

    print("Train/validation/test splits created.")
    print(f"Input rows: {summary['input_rows']}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Validation rows: {summary['valid_rows']}")
    print(f"Test rows: {summary['test_rows']}")
    print(f"Input fraud rate: {summary['input_fraud_rate']:.3%}")
    print(f"Train fraud rate: {summary['train_fraud_rate']:.3%}")
    print(f"Validation fraud rate: {summary['valid_fraud_rate']:.3%}")
    print(f"Test fraud rate: {summary['test_fraud_rate']:.3%}")
    print(f"No overlap: {summary['no_overlap']}")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()