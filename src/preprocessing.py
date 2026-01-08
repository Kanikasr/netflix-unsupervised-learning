import pandas as pd
import numpy as np
import re


PLACEHOLDERS = ['no data', 'n/a', 'na', 'unknown', 'none', '-', 'not specified']


def preprocess_netflix_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Normalize placeholders
    df.replace(PLACEHOLDERS, pd.NA, inplace=True)

    # Date handling
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year

    # Text columns
    for col in ['director', 'cast', 'country', 'rating', 'listed_in']:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    df['description'] = df['description'].fillna('').astype(str)

    # Duration parsing
    df['duration'] = df['duration'].astype(str)
    df['duration_int'] = df['duration'].str.extract(r'(\d+)', expand=False).astype(float)
    df['duration_type'] = (
        df['duration']
        .str.extract(r'([A-Za-z]+)', expand=False)
        .str.lower()
        .fillna('')
    )

    # Genre count
    df['num_genres'] = df['listed_in'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) else 0
    )

    # Delay years
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['delay_years'] = df['year_added'] - df['release_year']
    df['delay_years'] = df['delay_years'].where(df['delay_years'].between(0, 50))

    return df
