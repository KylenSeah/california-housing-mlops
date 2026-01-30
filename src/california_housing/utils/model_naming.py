"""
Model naming utilities.

This module provides a single source of truth for constructing registered
model names, ensuring consistency between the training pipeline (writer)
and the inference service (reader).
"""

from california_housing.core.config_definitions import MlflowConfig


def get_registered_model_name(mlflow_config: MlflowConfig, model_name: str) -> str:
    """
    Construct the canonical name for the MLflow Model Registry.

    Args:
        mlflow_config (MlflowConfig): Configuration containing the name prefix.
        model_name (str): The specific algorithm name (e.g. 'LGBMRegressor').

    Returns:
        str: The full registry name (e.g., 'california_housing-LGBMRegressor').
    """
    return f"{mlflow_config.registered_model_prefix}-{model_name}"
