import argparse
import json 

from app.models.batch_predict import batch_predict


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Run batch fraud-risk predictions."
    )

    parser.add_argument(
        "--input-path",
        default= "data/current/current_transactions.csv",
        help= "Pah to input transactions CSV.",
    )

    parser.add_argument(
        "--output-path",
        default= None,
        help="Optional output prediction CSV path.",
    )

    parser.add_argument(
        "--summary-path",
        default= None, 
        help= "Optional output summary JSON path."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summary = batch_predict(
        input_path= args.input_path,
        output_path= args.output_path,
        summary_path = args.summary_path,
    )

    print("Batch prediction completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()