"""
Utilities for data binning and categorization.

This module provides generic mathematical helpers for transforming continuous
variables into discrete categories, primarily for stratification purposes.
"""

import pandas as pd


def create_stratification_bins(
    df: pd.DataFrame, col_name: str, bins: list[float], labels: list[int]
) -> pd.Series:
    """
    Discretize a continuous variable into defined buckets.

    Used to create a categorical proxy for a numerical feature to enable
    stratified sampling.

    Args:
        df (pd.DataFrame): The source DataFrame.
        col_name (str): The numerical column to bin.
        bins (List[float]): The boundary values for the buckets.
        labels (List[int]): The identifiers for each bucket.

    Returns:
        pd.Series: A categorical series representing the assigned bins.
    """
    return pd.cut(x=df[col_name], bins=bins, labels=labels)
