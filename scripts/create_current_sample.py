from pathlib import Path 

from app.data.ingest import load_transactions


def main():
    input_path = Path("data/synthetic/transactions.csv")
    output_path = Path("data/current/current_transactions.csv")

    df = load_transactions(input_path)

    sample_df = df.sample(
        n=min(100, len(df)),
        random_state= 42,
    ).drop(columns=['is_fraud'], errors= 'ignore')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index= False)

    print(f'Saved current sample to: {output_path}')


if __name__ == "__main__":
    main()