import argparse 
import json 

from app.monitoring.retraining import run_retraining_trigger_check


def parse_args():
    parser = argparse.ArgumentParser(
        description= 'Check whether model retraining should be triggered'
    )

    parser.add_argument(
        '--drift-summary-path',
        default= 'data/results/drift_summary.json'
    )

    parser.add_argument(
        '--performance-summary-path',
        default= 'data/results/evaluation_metrics.json'
    )

    parser.add_argument(
        '--output-path',
        default= 'data/results/retraining_trigger_report.json'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    decision = run_retraining_trigger_check(
        drift_summary_path= args.drift_summary_path,
        performance_summary_path= args.performance_summary_path,
        output_path= args.output_path,
    )

    print('Retraining trigger check completed')
    print(json.dumps(decision, indent=2))


if __name__ ==  '__main__':
    main()