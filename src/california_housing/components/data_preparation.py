"""
Data preparation and partitioning component.

This module provides logic for splitting raw DataFrames into training and
testing sets. It implements stratified sampling to ensure that the
distributions of critical features (e.g., income) are preserved across splits.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from california_housing.core.config_definitions import (
    DataPreparationConfig,
    GlobalsConfig,
)
from california_housing.core.exceptions import DataValidationError
from california_housing.utils.binning import create_stratification_bins

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainTestSplit:
    """
    Immutable container for partitioned dataset features and targets.

    Attributes:
        X_train_full: Features for the training set (including validation data).
        y_train_full: Target values for the training set.
        X_test: Features isolated for final model evaluation.
        y_test: Target values isolated for final model evaluation.
    """

    X_train_full: pd.DataFrame
    y_train_full: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class DataPreparation:
    """
    Orchestrate the partitioning of the dataset into train and test sets.

    This component is responsible for isolating the test set immediately after
    ingestion to prevent data leakage. It supports both random and stratified
    splitting strategies based on the provided configuration.
    """

    def __init__(
        self, config: DataPreparationConfig, globals_config: GlobalsConfig
    ) -> None:
        """
        Initialize the DataPreparation component.

        Args:
            config (DataPreparationConfig): Configuration settings for
                preprocessing and split proportions.
            globals_config (GlobalsConfig): Global settings including
                random state and target column identifier.
        """
        self.config = config
        self.globals = globals_config

    def split_data(self, df: pd.DataFrame) -> TrainTestSplit:
        """
        Partition the input DataFrame into training and testing sets.

        Executes logic to verify column existence, generate stratification bins
        if enabled, and perform the physical split using Scikit-Learn.

        Args:
            df (pd.DataFrame): The raw, validated DataFrame from the ingestion layer.

        Raises:
            ValueError: If the input is empty or required columns are missing.
            DataValidationError: If stratification fails due to out-of-bounds
                values or insufficient samples per bin.

        Returns:
            TrainTestSplit: An object containing the four split partitions.
        """
        logger.info("Splitting data and isolating test set...")

        if df.empty:
            raise ValueError("Input DataFrame is empty. Pipeline logic error.")

        target_col = self.globals.target_col
        stratify_on = self.config.preprocessing.stratify_on_col

        # 1. Verification Logic
        required_col = {target_col}
        if stratify_on:
            required_col.add(stratify_on)

        missing_col = required_col - set(df.columns)
        if missing_col:
            raise ValueError(f"Missing required columns: {missing_col}")

        # 2. Stratification Logic (The "Magic" Step)
        stratify_series = None
        if stratify_on:
            logger.info("Stratified splitting enabled on column: %s", stratify_on)

            stratify_series = create_stratification_bins(
                df=df,
                col_name=stratify_on,
                bins=self.config.preprocessing.stratify_bins,
                labels=self.config.preprocessing.stratify_labels,
            )

            # Safety Gate: Handle values outside the specified bins
            if stratify_series.isna().any():
                nan_counts = stratify_series.isna().sum()
                raise DataValidationError(
                    f"Stratification failed: {nan_counts} values in '{stratify_on}' "
                    f"fall outside defined bins: {self.config.preprocessing.stratify_bins}"
                )

            # Safety Gate: Ensure bins are large enough for a split
            bin_counts = stratify_series.value_counts()
            min_samples = bin_counts.min()

            if min_samples < 2:
                raise DataValidationError(
                    f"Stratification impossible. The smallest bin only has {min_samples} samples. "
                    f"StratifiedKFold requires at least 2 samples per group. "
                    f"Distribution:\n{bin_counts.to_dict()}"
                )

            # Safety Gate: Enforce user-defined minimum samples for robustness
            min_config_samples = self.config.preprocessing.min_stratify_samples
            if min_samples < min_config_samples:
                raise DataValidationError(
                    f"Insufficient samples for robust splitting. "
                    f"Found min {min_samples}, but config requires {min_config_samples}. "
                    f"Distribution:\n{bin_counts.to_dict()}"
                )
        else:
            logger.info("Stratification disabled. Performing random split.")

        # 3. Physical Split Execution
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()

        logger.info("Splitting data with test_size=%.2f", self.config.split.test_size)

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            stratify=stratify_series,
            test_size=self.config.split.test_size,
            random_state=self.globals.random_state,
        )

        logger.info(
            "Data split complete. X_train: %s, X_test: %s, y_train: %s, y_test: %s",
            X_train_full.shape,
            X_test.shape,
            y_train_full.shape,
            y_test.shape,
        )

        return TrainTestSplit(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
        )
