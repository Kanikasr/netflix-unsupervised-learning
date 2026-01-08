import pandas as pd


def load_netflix_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Netflix dataset from CSV.
    """
    df = pd.read_csv(filepath)
    return df
