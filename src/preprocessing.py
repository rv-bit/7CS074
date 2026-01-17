import csv
import os

import pandas as pd
import numpy as np

# We will globally define the data paths here, as they will be used in multiple functions / notebooks
# Base Directory for the project should be /, i.e., the parent directory of src
BASE_DIR = os.getcwd()

DATA_RAW_PATH = os.path.join(BASE_DIR, "data/raw")
DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data/clean")

DATASET_CLEAN_FILE_PATH = os.path.join(DATA_CLEAN_PATH, "cleaned_dataset.csv")

COLUMNS = [
    "make",
    "model",
    "year",
    "price",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize",
]

NUMERIC_OUTLIER_COLUMNS = [
    "price",
    "mileage",
    "mpg",
    "engineSize",
    "tax"
]

EXTRA_COLUMNS = ['mileage2', 'fuel type2', 'engine size2', 'reference']
EXPECTED_COLUMNS = 9

def process_raw_multiple_data_files():
    if not os.path.exists(DATA_RAW_PATH):
        raise FileNotFoundError("Data directory not found. Please ensure the project structure is correct.")

    # Creates if not exists
    if not os.path.exists(DATA_CLEAN_PATH):
        os.makedirs(DATA_CLEAN_PATH)

    if os.path.exists(DATASET_CLEAN_FILE_PATH):
        return  # If cleaned dataset already exists, skip processing

    # Creates if not exists, double check at this point
    if not os.path.exists(DATASET_CLEAN_FILE_PATH):
        os.makedirs(DATA_CLEAN_PATH, exist_ok=True)

    directory_raw_bytes = os.fsencode(DATA_RAW_PATH)

    # Validate that CSV files to see if exists and is non-empty
    for file in os.listdir(directory_raw_bytes):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            DATASET_PATH = os.path.join(DATA_RAW_PATH, filename)
            df = pd.read_csv(DATASET_PATH)
            if df.empty:
                raise ValueError("Loaded dataset is empty. Please check the dataset file.")
            continue
        else:
            continue

    # Concatenate all CSV files in the raw data directory
    # We will read the clean path / file directly, and add the default headers
    # We will use our COLUMNS variable as the schema
    schema = COLUMNS
    with open(DATASET_CLEAN_FILE_PATH, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([g for g in schema])

    for file in os.listdir(directory_raw_bytes):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            DATASET_PATH = os.path.join(DATA_RAW_PATH, filename)
            df_default = pd.read_csv(DATASET_PATH)
            df = clean_data(df_default)

            if not df.empty:
                make = filename.replace('.csv', '')
                df.insert(0, 'make', make) # Insert 'make' as first column
                
                # Only keep rows with exactly EXPECTED_COLUMNS + 1 columns after inserting 'make'
                df = df[df.apply(lambda x: len(x) == EXPECTED_COLUMNS + 1, axis=1)]

                df.to_csv(
                    DATASET_CLEAN_FILE_PATH,
                    mode='a',
                    header=False,
                    index=False
                )
        else:
            continue

def remove_iqr_outliers(
    df: pd.DataFrame,
    columns: list,
    factor: float = 1.5
) -> pd.DataFrame:
    """
        Remove outliers using Interquartile Range (IQR) method for selected columns.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

def apply_domain_constraints(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
        Apply domain constraints to the dataframe, meaning that we filter the data based on known valid ranges for each numeric column, so that we remove any rows that have values outside these ranges.
    """
    df = df.copy()

    constraints = {
        'price': lambda x: (x > 100) & (x < 200_000),
        'mileage': lambda x: (x >= 0) & (x < 300_000),
        'engineSize': lambda x: (x > 0) & (x < 12.0),
        'mpg': lambda x: (x > 5) & (x < 60),
        'tax': lambda x: (x >= 0)
    }

    for col, condition in constraints.items():
        if col in df.columns:
            df = df[condition(df[col])]

    return df

def coerce_numeric_columns(
    df: pd.DataFrame, 
    columns: list
) -> pd.DataFrame:
    """
        Here we make sure that any columns parse we parse to integers / numeric values, if record in column is string of currency numeric, we remove this regex
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[£,]", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# Cleans the data, and returns copy of cleaned dataframe
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Here we clean out the data from any possible problems which can cause bad training
    """
    df = df.copy()
    
    # Remove extra columns and duplicate columns if present
    df.drop(columns=[col for col in EXTRA_COLUMNS if col in df.columns], inplace=True, errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Only keep rows with exactly EXPECTED_COLUMNS columns
    df = df[df.apply(lambda x: len(x) == EXPECTED_COLUMNS, axis=1)]
    
    # Replace 'N/A' and Drop record duplicates
    # If by any chance 'N/A' is a string by text, we shall change it to nan 'na', later we will remove anyway
    df.replace("N/A", np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    # Drop rows that are all empty or just commas 
    # """
    #     ,,,,,,,,,,
    # """
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.dropna(how='all', inplace=True)

    # Coerce numeric columns
    df = coerce_numeric_columns(df, NUMERIC_OUTLIER_COLUMNS)
    # Domain filtering
    df = apply_domain_constraints(df)
    # Statistical outliers
    df = remove_iqr_outliers(
        df,
        columns=[c for c in NUMERIC_OUTLIER_COLUMNS if c in df.columns]
    )
    
    return df