
"""
Main Preprocessing Pipeline
Project: Logistics Data Preprocessing
Week 2 Internship Task

Runs the complete logistics data preprocessing workflow.
"""

import argparse
import os
import pandas as pd

from src.data_collection import load_dataset
from src.preprocessing import preprocess_data, save_processed_data


# Project paths
RAW_DATA_PATH = "data/raw/DataCoSupplyChainDataset.csv"
PROCESSED_DIR = "data/processed"

OUTPUT_DATA_PATH = os.path.join(
    PROCESSED_DIR,
    "logistics_cleaned.csv"
)

OUTLIER_REPORT_PATH = os.path.join(
    PROCESSED_DIR,
    "outlier_report.csv"
)


# Numerical measurement columns suitable for scaling.
# Identifiers, dates, and classification targets are excluded.
SCALING_CANDIDATES = [
    "days_for_shipping_real",
    "days_for_shipment_scheduled",
    "benefit_per_order",
    "sales_per_customer",
    "order_item_discount",
    "order_item_discount_rate",
    "order_item_product_price",
    "order_item_profit_ratio",
    "order_item_quantity",
    "sales",
    "order_item_total",
    "order_profit_per_order",
    "product_price"
]


def select_scaling_columns(df):
    """Select available continuous numerical features."""

    selected = [
        column
        for column in SCALING_CANDIDATES
        if column in df.columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not selected:
        raise ValueError(
            "No eligible numerical columns found. "
            "Check the dataset column names."
        )

    return selected


def run_pipeline(
    input_path=RAW_DATA_PATH,
    scaling_method="minmax",
    apply_outlier_capping=False
):
    """Execute the complete preprocessing pipeline."""

    print("=" * 60)
    print("LOGISTICS DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load dataset
    print("\nSTEP 1: DATA COLLECTION")

    df = load_dataset(input_path)

    print("\nOriginal dataset shape:", df.shape)

    # Step 2: Standardize column names before feature selection
    print("\nSTEP 2: FEATURE SELECTION")

    from src.data_cleaning import standardize_column_names

    standardized_df = standardize_column_names(df)

    scaling_columns = select_scaling_columns(
        standardized_df
    )

    print("Selected numerical features:")
    print(scaling_columns)

    # Step 3: Run preprocessing
    print("\nSTEP 3: PREPROCESSING")

    processed_df, outlier_report = preprocess_data(
        df,
        scaling_method=scaling_method,
        scaling_columns=scaling_columns,
        apply_outlier_capping=apply_outlier_capping
    )

    # Step 4: Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 5: Export processed dataset
    print("\nSTEP 4: EXPORT PROCESSED DATA")

    save_processed_data(
        processed_df,
        OUTPUT_DATA_PATH
    )

    # Step 6: Export outlier report
    outlier_report.to_csv(
        OUTLIER_REPORT_PATH,
        index=True
    )

    print(
        f"Outlier report saved to: {OUTLIER_REPORT_PATH}"
    )

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)

    print("Original rows:", len(df))
    print("Processed rows:", len(processed_df))
    print("Original columns:", len(df.columns))
    print("Processed columns:", len(processed_df.columns))
    print("Scaling method:", scaling_method)

    return processed_df, outlier_report


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Logistics Data Preprocessing Pipeline"
    )

    parser.add_argument(
        "--input",
        default=RAW_DATA_PATH,
        help="Path to the original logistics CSV dataset"
    )

    parser.add_argument(
        "--scaling",
        choices=["minmax", "standard", "none"],
        default="minmax",
        help="Normalization method"
    )

    parser.add_argument(
        "--cap-outliers",
        action="store_true",
        help="Apply IQR-based outlier capping"
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        scaling_method=args.scaling,
        apply_outlier_capping=args.cap_outliers
    )

