"""
Model evaluation component.

This module is responsible for calculating performance metrics (RMSE, MAE, R2)
and persisting them to JSON artifacts. It acts as a standardized "ruler"
to measure model quality during training and testing.
"""

import json
import logging
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from california_housing.core.config_definitions import (
    ArtifactsConfig,
    TrainerEvaluationConfig,
)

logger = logging.getLogger(__name__)

# Registry of supported metrics mapping names to Scikit-Learn functions
METRIC_FUNCTIONS: dict[str, Callable] = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "mse": mean_squared_error,
}


class ModelEvaluator:
    """
    Compute and persist model performance metrics.

    This class decouples evaluation logic from training logic. It ensures
    that metrics are calculated consistently across different folds (CV)
    and different datasets (Validation vs Test).
    """

    def __init__(
        self, eval_config: TrainerEvaluationConfig, artifacts_config: ArtifactsConfig
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            eval_config (TrainerEvaluationConfig): Configuration defining which
                metrics to calculate (e.g. ['rmse', 'r2']).
            artifacts_config (ArtifactsConfig): Configuration defining where
                to save the metric JSON files.
        """
        self.eval_config = eval_config
        self.artifacts_config = artifacts_config

    def evaluate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str,
    ) -> dict[str, float]:
        """
        Run inference on the dataset and calculate all configured metrics.

        Args:
            pipeline (Pipeline): The fitted Scikit-Learn pipeline.
            X (pd.DataFrame): Features.
            y (pd.Series): Ground truth targets.
            set_name (str): Identifier for logging (e.g., 'validation', 'test').

        Raises:
            ValueError: If input data is empty, mismatched, or if the model
                produces NaNs/Infs.

        Returns:
            dict[str, float]: Dictionary of metric names and their values.
        """
        logger.info("--- Starting model evaluation on '%s' dataset ---", set_name)

        # 1. Validation: Ensure we support what is asked
        requested = set(self.eval_config.metrics)
        supported = set(METRIC_FUNCTIONS.keys())
        unsupported_metrics = requested - supported

        if unsupported_metrics:
            raise ValueError(
                f"Configuration Error: The following metrics are not supported: {sorted(unsupported_metrics)}."
                f"Available options: {sorted(supported)}"
            )

        if X.empty or y.empty:
            raise ValueError(f"Evaluation data for '{set_name}' is empty.")

        if len(X) != len(y):
            raise ValueError(
                f"Feature and target for '{set_name}' have mismatched lengths."
            )

        # 2. Inference
        predictions = pipeline.predict(X)

        # 3. Safety Check: Exploding Gradients / Bad Math
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise ValueError(
                f"Model produced NaNs or Infs on '{set_name}' set. "
                "Check preprocessing scalers, imputation logic or model hyperparameters."
            )

        # 4. Calculation (Strategy Pattern)
        metrics: dict[str, float] = {}
        for metrics_name in self.eval_config.metrics:
            # 2. Execution (Strategy Pattern)
            func = METRIC_FUNCTIONS[metrics_name]
            score = func(y, predictions)

            metrics[metrics_name] = float(score)

        # 5. Logging
        formatted_metrics = " | ".join(
            f"{k.upper()}: {v:.4f}" for k, v in metrics.items()
        )
        logger.info("Metrics for '%s' set: %s", set_name, formatted_metrics)

        # Return Structured Object
        return metrics

    def save_metrics(
        self,
        metrics: dict[str, float],
        model_name: str,
        set_name: str,
        run_id: str,
    ) -> Path:
        """
        Persist metrics to a JSON file in the artifacts directory.

        Args:
            metrics (dict): The dictionary of calculated scores.
            model_name (str): Name of the model (e.g. 'LGBMRegressor').
            set_name (str): Context (e.g. 'test').
            run_id (str): Unique identifier (e.g. MLflow Run ID or Timestamp).

        Returns:
            Path: The absolute path to the saved JSON file.
        """
        base_filename = self.eval_config.metrics_filename

        # Sanity check format string from config
        if "{model_name}" not in base_filename:
            raise ValueError(
                f"Invalid Configuration: 'metric_filename' ({base_filename}) "
                "must contain the placeholder '{model_name}' to ensure "
                "unique artifact generation for each model."
            )

        # Construct filename: e.g., "test_LGBMRegressor_metrics_a1b2c3d4.json"
        clean_filename = base_filename.format(model_name=model_name)
        filename = f"{set_name}_{clean_filename.replace('.json', '')}_{run_id[:8]}.json"

        metrics_dir = self.artifacts_config.root_dir / self.artifacts_config.metrics_dir
        metrics_path = metrics_dir / filename

        # Ensure directory exists
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        # Cast to strict Path for type checkers
        return cast(Path, metrics_path)
