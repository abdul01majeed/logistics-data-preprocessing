
"""
Data Collection Module
Project: Logistics Data Preprocessing
Week 2 Internship Task

This module handles downloading and loading the
DataCo Smart Supply Chain dataset.
"""

import os
import pandas as pd
import urllib.request


# Dataset configuration
DATASET_URL = (
    "https://data.mendeley.com/public-files/"
    "datasets/8gx2fvg2k6/files/"
)

RAW_DATA_DIR = "data/raw"
DATASET_FILE = "DataCoSupplyChainDataset.csv"


def create_data_directory():
    """Create the raw data directory if it does not exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Data directory ready: {RAW_DATA_DIR}")


def load_dataset(file_path):
    """
    Load a CSV dataset using Pandas.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    try:
        df = pd.read_csv(
            file_path,
            encoding="latin-1"
        )

        print("Dataset loaded successfully!")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    except Exception as error:
        raise RuntimeError(
            f"Error loading dataset: {error}"
        )


def collect_dataset():
    """
    Prepare the data directory and load the dataset
    if it has already been downloaded.
    """
    create_data_directory()

    file_path = os.path.join(
        RAW_DATA_DIR,
        DATASET_FILE
    )

    if os.path.exists(file_path):
        print("Existing dataset found.")
        return load_dataset(file_path)

    print(
        "Dataset is not downloaded yet. "
        "Please download the CSV from the official dataset "
        "source and place it inside data/raw/."
    )

    return None


def explore_dataset(df):
    """Display basic information about the dataset."""

    if df is None:
        print("No dataset available for exploration.")
        return

    print("\n--- DATASET PREVIEW ---")
    print(df.head())

    print("\n--- DATASET INFORMATION ---")
    print(df.info())

    print("\n--- STATISTICAL SUMMARY ---")
    print(df.describe(include="all"))

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())


if __name__ == "__main__":

    dataset = collect_dataset()

    if dataset is not None:
        explore_dataset(dataset)

