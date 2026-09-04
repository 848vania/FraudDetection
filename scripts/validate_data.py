import argparse
import sys

from app.data.validation import (
    load_raw_data,
    run_data_validation,
    save_validation_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Validate raw fraud-risk transaction data"
    )

    parser.add_argument(
        "--input-path",
        default= "data/raw/transactions.csv",
        help = "Path to raw transaction CSV",
    )

    parser.add_argument(
        "--output-path",
        default= "data/processed/validation_report.json",
        help= "Path where validation report JSON will be saved."
    )

    parser.add_argument(
        "--fail-on-warning",
        action= "store_true",
        help= "If set, warnings also cause the script to exit with code 1."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_raw_data(args.input_path)
    report = run_data_validation(df)

    save_validation_report(report, args.output_path)

    print("Validation completed.")
    print(f"Rows: {report['num_rows']}")
    print(f"Columns: {report['num_columns']}")
    print(f"Passed: {report['passed']}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")

    if report['errors']:
        print("\nErrors:")
        for error in report['errors']:
            print(f"- {error}")

    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"- {warning}")

    if not report['passed']:
        sys.exit(1)

    if args.fail_on_warning and report['warnings']:
        sys.exit(1)


if __name__ == "__main__":
    main()