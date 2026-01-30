import os

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

CATEGORICAL_FEATURES_GLOBAL = [
    'make',
    'model',
    'transmission',
    'fuelType'
]

CATEGORICAL_FEATURES_PER_MAKE = [
    'model',
    'transmission',
    'fuelType'
]

NUMERIC_FEATURES = [
    'year', 
    'tax', 
    'mileage',
    'engineSize',
    'mpg',
    
    # We are making this based on the (mpg/engine size)
    'efficiency_score',
    # We are making this base on (current_year - year)
    'vehicle_age'
]

CLUSTERING_FEATURES = [
    'price', 
    'mileage', 
    'mpg', 
    'engineSize', 
    'year'
]

ASSOCIATION_RULES = [
    'make', 
    'fuelType',
    'transmission',
    'price',
    'mileage',
    'engineSize',
    'mpg'
]

EXTRA_COLUMNS = ['mileage2', 'fuel type2', 'engine size2', 'reference']
EXPECTED_COLUMNS = 9