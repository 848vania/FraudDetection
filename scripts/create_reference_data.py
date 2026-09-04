from pathlib import Path

from app.data.ingest import load_transactions


def main():
    input_path = Path('data/processed/train.csv')
    output_path = Path('data/reference/reference_transactions.csv')

    df = load_transactions(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f'Saved reference data to {output_path}')


if __name__ == "__main__":
    main()