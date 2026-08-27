import argparse 
import json 

from app.monitoring.drift import run_drift_check


def parse_args():
    parser = argparse.ArgumentParser(
        description= "Run data and prediction drift checks"
    )

    parser.add_argument(
        "--reference-path",
        default= 'data/reference/reference_transactions.csv',
        help = "Path to reference transaction CSV",
    )

    parser.add_argument(
        "--current-path",
        default= "data/current/current_transactions.csv",
        help = "Path to current transaction CSV",
    )

    parser.add_argument(
        "--reference-predictions-path",
        default= None,
        help = "Optional reference predictions CSV",
    )

    parser.add_argument(
        "--current-predictions-path",
        default= None,
        help = "Optional current predictions CSV",
    )

    parser.add_argument(
        "--output-summary-path",
        default= "data/results/drift_summary.json",
        help = "Path to output drift summary JSON",
    )

    parser.add_argument(
        "--output-report-path",
        default= "data/results/drift_report.html",
        help = "Path to output HTML drift report",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summary = run_drift_check(
        reference_path= args.reference_path,
        current_path= args.current_path,
        output_summary_path= args.output_summary_path,
        output_report_path= args.output_report_path,
        reference_predictions_path= args.reference_predictions_path,
        current_predictions_path= args.current_predictions_path,
    )

    print("Drift check completed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()