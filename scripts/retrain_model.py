import argparse 
import json 

from app.models.retrain import run_retraining_pipeline
from app.models.train import  load_training_config


def parse_args():
    parser = argparse.ArgumentParser(
        description= 'Run automated model retraining pipeline'
    )

    parser.add_argument(
        '--config-path',
        default= 'configs/retraining_logistic.yaml',
    )

    parser.add_argument(
        '--trigger-report-path',
        default= 'data/results/retraining_trigger_report.json',
    )

    parser.add_argument(
        '--champion-evaluation-path',
        default= 'data/results/evaluation_metrics.json',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_training_config(args.config_path)

    report = run_retraining_pipeline(
        retraining_config= config, 
        trigger_report_path= args.trigger_report_path,
        champion_evaluation_path= args.champion_evaluation_path,
    )

    print('Retraining pipeline completed.')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()