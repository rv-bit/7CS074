import pandas as pd
import csv
import os

def process_raw_multiple_data_files():
    # Base Directory for the project should be /, i.e., the parent directory of src
    BASE_DIR = os.getcwd()

    DATA_RAW_PATH = os.path.join(BASE_DIR, "data/raw")
    DATA_CLEAN_PATH = os.path.join(BASE_DIR, "data/clean")

    if not os.path.exists(DATA_RAW_PATH):
        raise FileNotFoundError("Data directory not found. Please ensure the project structure is correct.")

    # Creates if not exists
    if not os.path.exists(DATA_CLEAN_PATH):
        os.makedirs(DATA_CLEAN_PATH)

    DATASET_CLEAN_PATH = os.path.join(DATA_CLEAN_PATH, "cleaned_dataset.csv")

    if os.path.exists(DATASET_CLEAN_PATH):
        return  # If cleaned dataset already exists, skip processing

    # Creates if not exists, double check at this point
    if not os.path.exists(DATASET_CLEAN_PATH):
        os.makedirs(DATA_CLEAN_PATH, exist_ok=True)

    directory_raw_bytes = os.fsencode(DATA_RAW_PATH)

    # Validate that at least one CSV file exists and is non-empty
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
    # We will first read one of the raw files to get the schema
    # Then we will join extra column for the make of vehicle where the filename is the make of vehicle
    sample_file = os.path.join(DATA_RAW_PATH, os.listdir(directory_raw_bytes)[0].decode("utf-8"))
    schema = pd.read_csv(sample_file, nrows=0).columns.tolist()
    schema.insert(0, 'make') # Insert 'make' column at the start
    with open(DATASET_CLEAN_PATH, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([g for g in schema])

    for file in os.listdir(directory_raw_bytes):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            DATASET_PATH = os.path.join(DATA_RAW_PATH, filename)
            make = filename.replace('.csv', '')  # Extract make from filename
            df = pd.read_csv(DATASET_PATH)
            df.insert(0, 'make', make)  # Insert 'make' column
            df.to_csv(DATASET_CLEAN_PATH, mode='a', header=False, index=False)
        else:
            continue
