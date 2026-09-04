from pathlib import Path

import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    """
    Load transaction data from a CSV file 

    Parameters
    -----------
    path:
        Path to a transaction CSV file

    Returns 
    --------
    pd.DataFrame
        Loaded transaction data

    Raises 
    -------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the loaded dataframe is empty 
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Transaction data file not found: {path}")

    df  = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Transaction data file is empty: {path}")

    return df 


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save a dataframe to CSV.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Output CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_raw_transactions(
        path: str | Path = 'data/raw/transactions.csv',
) -> pd.DataFrame:
    """
    Convenience wrapper for loading the default raw transaction dataset
    """
    return load_transactions(path)