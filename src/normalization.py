
"""
Normalization Module
Project: Logistics Data Preprocessing
Week 2 Internship Task

Implements Min-Max Scaling and Standardization
using Scikit-learn.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_numeric_columns(df):
    """Return numerical columns suitable for scaling."""

    return df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


def min_max_normalization(df, columns=None):
    """
    Scale numerical features to a range of 0 to 1.

    Formula:
        X_scaled = (X - X_min) / (X_max - X_min)
    """

    df = df.copy()

    if columns is None:
        columns = get_numeric_columns(df)

    if not columns:
        return df

    scaler = MinMaxScaler()

    df[columns] = scaler.fit_transform(
        df[columns]
    )

    print("Min-Max normalization completed.")

    return df


def standardization(df, columns=None):
    """
    Standardize numerical features.

    Formula:
        Z = (X - mean) / standard deviation

    Result:
        Mean approximately 0
        Standard deviation approximately 1
    """

    df = df.copy()

    if columns is None:
        columns = get_numeric_columns(df)

    if not columns:
        return df

    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(
        df[columns]
    )

    print("Standardization completed.")

    return df


def compare_scaling_methods(df, columns=None):
    """Compare original, Min-Max, and standardized values."""

    if columns is None:
        columns = get_numeric_columns(df)

    if not columns:
        raise ValueError(
            "No numerical columns available for scaling."
        )

    comparison = pd.DataFrame(index=df.index)

    for column in columns:

        comparison[f"{column}_original"] = df[column]

        comparison[f"{column}_minmax"] = (
            MinMaxScaler().fit_transform(
                df[[column]]
            ).ravel()
        )

        comparison[f"{column}_standardized"] = (
            StandardScaler().fit_transform(
                df[[column]]
            ).ravel()
        )

    return comparison


if __name__ == "__main__":

    sample_data = pd.DataFrame({
        "shipping_cost": [10, 20, 30, 40, 50],
        "delivery_days": [2, 4, 6, 8, 10],
        "order_quantity": [1, 3, 5, 7, 9]
    })

    print("\nOriginal Data:")
    print(sample_data)

    print("\nMin-Max Scaled Data:")
    print(min_max_normalization(sample_data))

    print("\nStandardized Data:")
    print(standardization(sample_data))

