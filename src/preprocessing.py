
"""
Preprocessing Integration Module
Project: Logistics Data Preprocessing
Week 2 Internship Task

Combines data cleaning, outlier detection,
outlier treatment, and normalization.
"""

import pandas as pd

from src.data_cleaning import clean_dataset
from src.outlier_detection import detect_outliers_iqr
from src.outlier_detection import cap_outliers_iqr
from src.normalization import min_max_normalization
from src.normalization import standardization


def preprocess_data(
    df,
    scaling_method="minmax",
    scaling_columns=None,
    apply_outlier_capping=False
):
    """
    Execute the logistics data preprocessing workflow.

    Args:
        df: Original Pandas DataFrame.
        scaling_method: 'minmax', 'standard' or 'none'.
        scaling_columns: Explicit numerical columns to scale.
        apply_outlier_capping: Whether to cap IQR outliers.

    Returns:
        cleaned_df: Final processed DataFrame.
        outlier_report: Outlier detection results.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if scaling_method not in ["minmax", "standard", "none"]:
        raise ValueError(
            "Invalid scaling method. Choose minmax, standard or none."
        )

    print("\nSTARTING LOGISTICS PREPROCESSING")
    print("=" * 50)

    # Step 1: Data cleaning
    print("\nSTEP 1: DATA CLEANING")

    cleaned_df = clean_dataset(df)

    # Step 2: Outlier detection
    print("\nSTEP 2: OUTLIER DETECTION")

    outlier_report = detect_outliers_iqr(
        cleaned_df,
        columns=scaling_columns
    )

    print(outlier_report)

    # Step 3: Optional outlier treatment
    if apply_outlier_capping:

        print("\nSTEP 3: OUTLIER CAPPING")

        cleaned_df = cap_outliers_iqr(
            cleaned_df,
            columns=scaling_columns
        )

    # Step 4: Normalization
    print("\nSTEP 4: DATA NORMALIZATION")

    if scaling_method == "minmax":

        cleaned_df = min_max_normalization(
            cleaned_df,
            columns=scaling_columns
        )

    elif scaling_method == "standard":

        cleaned_df = standardization(
            cleaned_df,
            columns=scaling_columns
        )

    else:
        print("Normalization skipped.")

    print("\nPREPROCESSING COMPLETED")
    print("=" * 50)

    return cleaned_df, outlier_report


def save_processed_data(df, output_path):
    """Save the processed dataset as a CSV file."""

    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":

    sample_data = pd.DataFrame({
        "shipping_cost": [10, 20, 30, 40, 50, 60],
        "delivery_days": [2, 3, 4, 5, 6, 7],
        "order_quantity": [1, 2, 3, 4, 5, 6]
    })

    processed_data, outlier_results = preprocess_data(
        sample_data,
        scaling_method="minmax"
    )

    print("\nFINAL PROCESSED DATA:")
    print(processed_data)

