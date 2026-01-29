"""
Model Training and Validation Component.

This module orchestrates the core machine learning workflow:
1. Constructing the Scikit-Learn pipeline (Preprocessing + Model).
2. executing the validation strategy (Single Split or Cross-Validation).
3. enforcing Quality Gates (threshold checks).
4. Retraining the final model on the full dataset.
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from california_housing.components.data_preparation import TrainTestSplit
from california_housing.components.data_transformer import create_preprocessing_pipeline
from california_housing.components.model_evaluator import ModelEvaluator
from california_housing.core.config_definitions import (
    DataPreparationConfig,
    GlobalsConfig,
    ModelTrainerConfig,
)
from california_housing.core.exceptions import ModelTrainingError
from california_housing.models.catalog import get_model_class
from california_housing.utils.binning import create_stratification_bins

logger = logging.getLogger(__name__)

METRIC_PREFIX_MEAN = "validation_mean_"
METRIC_PREFIX_STD = "validation_std_"


@dataclass(frozen=True)
class TrainingResult:
    """A structured container for the results of a model training run."""

    fitted_pipeline: Pipeline
    validation_metrics: dict[str, float]


class ModelTrainer:
    """
    Component for building, training, and validating a model pipeline.

    This class acts as the 'Conductor' of the training process. It doesn't
    know the specifics of the algorithm (it asks the Catalog) or the
    preprocessing details (it asks the Transformer Factory), but it knows
    how to combine them into a robust workflow.
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        globals_config: GlobalsConfig,
        data_prep_config: DataPreparationConfig,
        model_name: str,
        model_evaluator: ModelEvaluator,
    ) -> None:
        """
        Initialize the trainer with configuration and dependencies.

        Args:
            config (ModelTrainerConfig): Training-specific settings (strategy, params).
            globals_config (GlobalsConfig): Global settings (random_state).
            data_prep_config (DataPreparationConfig): Needed for stratification logic.
            model_name (str): The name of the model to train (e.g., 'LGBMRegressor').
            model_evaluator (ModelEvaluator): Service to calculate metrics.

        Raises:
            ValueError: If the requested model_name is not defined in the config params.
        """
        self.config = config
        self.globals_config = globals_config
        self.data_prep_config = data_prep_config
        self.model_name = model_name
        self.evaluator = model_evaluator

        if self.model_name not in self.config.params:
            raise ValueError(
                f"No parameters defined for model '{self.model_name}' in config."
            )

        self.model_params = self.config.params[self.model_name]

    def train(self, data: TrainTestSplit) -> TrainingResult:
        """
        Execute the full training lifecycle.

        Steps:
        1. Build the Pipeline.
        2. Validate (CV or Holdout).
        3. Check Quality Gate (Raise error if RMSE is too high).
        4. Retrain on full data (X_train_full).

        Args:
            data (TrainTestSplit): The dataset split into Train/Test. Note that
                validation splits are derived from 'X_train_full' inside this method.

        Raises:
            ValueError: If training data is empty.
            ModelTrainingError: If validation fails the quality threshold or crashes.

        Returns:
            TrainingResult: The final fitted pipeline and the validation metrics.
        """
        if data.X_train_full.empty:
            raise ValueError("Training data (X_train_full) is empty.")

        logger.info("Starting model training for %s", self.model_name)

        try:
            # 1. Build Base Pipeline
            base_pipeline = self._build_full_pipeline()

            # 2. Validation
            validation_metrics = self._perform_validation(
                data=data, base_pipeline=base_pipeline
            )

            # 3. Quality Gate
            primary_metric = self.config.evaluation.metrics[0]  # e.g. "rmse"
            metric_key = self._get_metric_key(primary_metric, METRIC_PREFIX_MEAN)

            current_score = validation_metrics[metric_key]

            threshold = self.config.quality_threshold

            if current_score > threshold:
                raise ModelTrainingError(
                    f"Quality Gate Failed: {primary_metric.upper()} {current_score:.4f} "
                    f"exceeds threshold {threshold}. Pipeline aborted to prevent "
                    f"deployment of sub-standard model."
                )

            # 4. Final Training (on all available training data)
            logger.info("Validation complete. Training final model...")
            base_pipeline.fit(data.X_train_full, data.y_train_full)
            logger.info("Final model training complete.")

            return TrainingResult(
                fitted_pipeline=base_pipeline, validation_metrics=validation_metrics
            )

        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {e}") from e

    def _build_full_pipeline(self) -> Pipeline:
        """Constructs the full scikit-learn pipeline with preprocessing and model."""
        # 1. Preprocessor
        preprocessor = create_preprocessing_pipeline(
            config=self.config.preprocessing,
            random_state=self.globals_config.random_state,
        )

        # 2. Model
        ModelClass = get_model_class(self.model_params.type)

        # Convert Pydantic model to dict, excluding the discriminator field
        model_params_dict = self.model_params.model_dump(exclude={"type"})

        # Inject global random_state if the model supports it
        sig = inspect.signature(ModelClass.__init__)
        if "random_state" in sig.parameters:
            model_params_dict["random_state"] = self.globals_config.random_state

        model = ModelClass(**model_params_dict)
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def _perform_validation(
        self,
        data: TrainTestSplit,
        base_pipeline: Pipeline,
    ) -> dict[str, float]:
        """
        Execute the configured validation strategy (Split vs CV).

        Args:
            data (TrainTestSplit): Data container.
            base_pipeline (Pipeline): Unfitted pipeline instance.

        Returns:
            dict[str, float]: Aggregated metrics (mean/std) from validation.
        """
        strategy = self.config.validation_strategy
        logger.info("Executing validation strategy: %s", strategy)

        # ---------------------------------------------------------------------
        # STRATIFICATION LOGIC (Config-Driven)
        # ---------------------------------------------------------------------
        stratify_on = self.data_prep_config.preprocessing.stratify_on_col
        stratify_col = None

        # Check if stratification is implemented
        if stratify_on:
            if stratify_on not in data.X_train_full.columns:
                raise ValueError(
                    f"Stratification Error: Configured column '{stratify_on}' "
                    "not found in X_train_full. Cannot perform stratified validation."
                )

            logger.info("Generating stratification bins on column: '%s'", stratify_on)

            stratify_col = create_stratification_bins(
                df=data.X_train_full,
                col_name=stratify_on,
                bins=self.data_prep_config.preprocessing.stratify_bins,
                labels=self.data_prep_config.preprocessing.stratify_labels,
            )
        else:
            # Stratification Disabled
            logger.info(
                "Stratification disabled (not configured). Using random splits."
            )

        # ---------------------------------------------------------------------
        # STRATEGY A: Single Split
        # ---------------------------------------------------------------------
        if strategy == "single_split":
            # Safety Assert: Config Loader guarantees this, but we check for sanity.
            assert self.config.single_split_config is not None, (
                "ModelTrainer received an invalid config state. "
                "The Pydantic validator in ModelTrainerConfig should have caught this."
            )

            X_train, X_val, y_train, y_val = train_test_split(
                data.X_train_full,
                data.y_train_full,
                test_size=self.config.single_split_config.val_size,
                stratify=stratify_col,
                random_state=self.globals_config.random_state,
            )

            # Clone to ensure we don't mutate the base instance
            pipeline_clone = clone(base_pipeline)
            pipeline_clone.fit(X_train, y_train)

            raw_metrics = self.evaluator.evaluate(
                pipeline=pipeline_clone,
                X=X_val,
                y=y_val,
                set_name="validation",
            )

            # Wrap in list to reuse the aggregation logic
            metrics = self._aggregate_cv_metrics([raw_metrics])
            self._log_validation_summary(metrics)

            return metrics

        # ---------------------------------------------------------------------
        # STRATEGY B: Cross Validation
        # ---------------------------------------------------------------------
        elif strategy == "cross_validation":
            assert self.config.cross_validation_config is not None, (
                "ModelTrainer received an invalid config state. "
                "The Pydantic validator in ModelTrainerConfig should have caught this."
            )

            n_folds = self.config.cross_validation_config.n_folds

            if stratify_col is not None:
                logger.info("Using StratifiedKFold with %s splits", n_folds)
                splitter = StratifiedKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=self.globals_config.random_state,
                )
            else:
                logger.info(
                    "Stratification disabled. Using Standard KFold with %s splits.",
                    n_folds,
                )
                splitter = KFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=self.globals_config.random_state,
                )

            all_fold_metrics: list[dict[str, float]] = []

            # Loop over folds
            for fold, (train_idx, val_idx) in enumerate(
                splitter.split(
                    X=data.X_train_full,
                    y=cast(Any, stratify_col),
                )
            ):
                logger.info("--- Starting CV Fold %s/%s ---", fold + 1, n_folds)

                pipeline_clone = clone(base_pipeline)

                X_fold_train = data.X_train_full.iloc[train_idx]
                y_fold_train = data.y_train_full.iloc[train_idx]
                X_fold_val = data.X_train_full.iloc[val_idx]
                y_fold_val = data.y_train_full.iloc[val_idx]

                pipeline_clone.fit(X_fold_train, y_fold_train)

                fold_metrics = self.evaluator.evaluate(
                    pipeline=pipeline_clone,
                    X=X_fold_val,
                    y=y_fold_val,
                    set_name=f"validation_fold_{fold + 1}",
                )
                all_fold_metrics.append(fold_metrics)

            metrics = self._aggregate_cv_metrics(all_fold_metrics)

            self._log_validation_summary(agg_metrics=metrics)

            return metrics

        else:
            raise NotImplementedError(f"Validation strategy '{strategy}' invalid.")

    def _aggregate_cv_metrics(
        self, all_fold_metrics: list[dict[str, float]]
    ) -> dict[str, float]:
        """
        Calculate Mean and Std Dev for metrics across all folds.

        Args:
            all_fold_metrics: List of metric dicts (one per fold).

        Returns:
            Dict with 'validation_mean_rmse', 'validation_std_rmse', etc.
        """
        if not all_fold_metrics:
            return {}

        agg_metrics: dict[str, float] = {}
        # takes the keys from first dictionary
        metric_keys = all_fold_metrics[0].keys()

        # Create list of all errors by type
        for key in metric_keys:
            values = [fold[key] for fold in all_fold_metrics]

            mean_key = self._get_metric_key(key, METRIC_PREFIX_MEAN)
            std_key = self._get_metric_key(key, METRIC_PREFIX_STD)

            agg_metrics[mean_key] = float(np.mean(values))
            agg_metrics[std_key] = float(np.std(values))

        return agg_metrics

    def _log_validation_summary(self, agg_metrics: dict[str, float]) -> None:
        """
        Log the primary metric statistics to the console for quick feedback.

        Args:
            agg_metrics (dict[str, float]): Dictionary containing mean and std
                keys for the evaluated metrics.

        Raises:
            KeyError: If the expected metric keys (mean/std) are missing from
                the aggregation result.
        """
        primary_metric = self.config.evaluation.metrics[0]

        mean_key = self._get_metric_key(primary_metric, METRIC_PREFIX_MEAN)
        std_key = self._get_metric_key(primary_metric, METRIC_PREFIX_STD)

        try:
            mean_score = agg_metrics[mean_key]
            std_score = agg_metrics[std_key]
        except KeyError as e:
            raise KeyError(
                f"Metric Contract Violation: Expected key '{e}' not found in results. "
                f"Available keys: {list(agg_metrics.keys())}. "
                "Check 'METRIC_PREFIX' or naming logic."
            ) from e

        logger.info(
            "CV Validation Summary: %s = %.4f ± %.4f",
            primary_metric.upper(),
            mean_score,
            std_score,
        )

    def _get_metric_key(self, metric_name: str, prefix: str) -> str:
        """Format the metric key consistently (e.g., 'validation_mean_rmse')."""
        return f"{prefix}{metric_name}"
