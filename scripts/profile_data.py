import argparse 

from app.data.ingest import load_transactions
from app.data.profiling import generate_data_profile, save_data_profile


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Generate a data profile for the fraud-risk dataset."
    )

    parser.add_argument(
        "--input-path",
        default= "data/raw/transactions.csv",
        help = "Path to raw transaction CSV",
    )

    parser.add_argument(
        "--output-path",
        default= "data/processed/data_profile.json",
        help = "Path where data profile JSON will be saved."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_transactions(args.input_path)
    profile = generate_data_profile(df)
    save_data_profile(profile, args.output_path)

    print("Data profile generated")
    print(f"Rows: {profile['num_rows']}")
    print(f"Columns: {profile['num_columns']}")
    print(f"Fraud rate: {profile.get('fraud_rate', 0):.2%}")
    print(
        "Majority-class baseline accuracy: "
        f"{profile.get('majority_class_accuracy', 0):.2%}"
    )
    print(f"Saved to: {args.output_path}")

    print("\nProfile notes:")
    for note in profile.get('profile_notes', []):
        print(f"- {note}")


if __name__ == "__main__":
    main()