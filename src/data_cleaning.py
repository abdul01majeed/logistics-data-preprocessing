
"""
Data Cleaning Module
Project: Logistics Data Preprocessing
Week 2 Internship Task

Handles missing values, duplicate records,
column names, and data type corrections.
"""

import pandas as pd


def standardize_column_names(df):
    """Convert column names to lowercase snake_case."""

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def analyze_missing_values(df):
    """Calculate missing values and their percentages."""

    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        )
    })

    return missing.sort_values(
        by="missing_percentage",
        ascending=False
    )


def handle_missing_values(df):
    """
    Handle missing values using appropriate strategies.

    Numerical columns: median imputation.
    Categorical columns: mode imputation.
    """

    df = df.copy()

    for column in df.columns:

        if df[column].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

        else:

            mode_values = df[column].mode(
                dropna=True
            )

            if not mode_values.empty:
                df[column] = df[column].fillna(
                    mode_values.iloc[0]
                )

    return df


def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""

    initial_rows = len(df)

    df = df.drop_duplicates().copy()

    removed_rows = initial_rows - len(df)

    print(f"Duplicate rows removed: {removed_rows}")

    return df


def convert_data_types(df):
    """
    Convert recognized logistics date columns
    into Pandas datetime format.
    """

    df = df.copy()

    date_columns = [
        column for column in df.columns
        if "date" in column or column.endswith("_at")
    ]

    for column in date_columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    return df


def clean_dataset(df):
    """Execute the complete data cleaning process."""

    print("Starting data cleaning...")

    initial_rows = len(df)

    # Standardize column names
    df = standardize_column_names(df)

    # Analyze missing values
    print("\nMissing values before cleaning:")
    print(analyze_missing_values(df))

    # Remove duplicate rows
    df = remove_duplicates(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Convert date columns
    df = convert_data_types(df)

    print("\nData cleaning completed.")

    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {len(df)}")

    print("\nMissing values after cleaning:")
    print(analyze_missing_values(df))

    return df

