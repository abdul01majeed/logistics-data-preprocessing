
"""
Outlier Detection Module
Project: Logistics Data Preprocessing
Week 2 Internship Task

Identifies numerical outliers using the
Interquartile Range (IQR) method.
"""

import pandas as pd


def detect_outliers_iqr(df, columns=None):
    """
    Detect outliers using the IQR method.

    Lower bound = Q1 - 1.5 * IQR
    Upper bound = Q3 + 1.5 * IQR

    Returns:
        Dictionary containing outlier statistics.
    """

    if columns is None:
        columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

    results = {}

    for column in columns:

        if column not in df.columns:
            print(f"Column not found: {column}")
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        values = df[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = values[
            (values < lower_bound) |
            (values > upper_bound)
        ]

        results[column] = {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers),
            "outlier_percentage": (
                len(outliers) / len(values) * 100
            )
        }

    return pd.DataFrame(results).T


def cap_outliers_iqr(df, columns=None):
    """
    Apply IQR-based capping (winsorization).

    Values below the lower bound or above the
    upper bound are capped at the respective bound.

    Use only after investigating whether extreme
    values are errors or valid observations.
    """

    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

    for column in columns:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[column] = df[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return df


def summarize_outliers(df):
    """Print an outlier detection summary."""

    results = detect_outliers_iqr(df)

    print("\nOUTLIER DETECTION SUMMARY")
    print("=" * 50)

    if results.empty:
        print("No numerical columns available.")
    else:
        print(results)

    return results

