import argparse

from app.data.ingest import load_transactions
from app.features.build_features import (
    build_feature_dataframe,
    save_feature_dataframe,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Build feature dataframe for fraud-risk modeling"
    )

    parser.add_argument(
        "--input-path",
        default= "data/raw/transactions.csv",
        help= "Path to raw transaction CSV",
    )

    parser.add_argument(
        "--output-path",
        default= "data/processed/features.csv",
        help = "Path where feature CSV will be saved."
    )

    parser.add_argument(
        "--include-target",
        action= "store_true",
        default= True,
        help= "Include target coclumn in output if present"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    raw_df = load_transactions(args.input_path)
    feature_df = build_feature_dataframe(
        raw_df,
        include_target= args.include_target,
    )

    save_feature_dataframe(feature_df, args.output_path)

    print("Feature dataframe created.")
    print(f"Input rows: {len(raw_df)}")
    print(f"Output rows: {len(feature_df)}")
    print(f"Output columns: {len(feature_df.columns)}")
    print(f"Saved to: {args.output_path}")

    print("\nCreate columns:")
    for column in feature_df.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()