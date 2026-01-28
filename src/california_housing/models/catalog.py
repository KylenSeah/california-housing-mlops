"""
Central registry for supported machine learning model architectures.

This module provides a factory-pattern interface to map string identifiers
from the configuration files to concrete Scikit-Learn compatible classes.
It ensures the training pipeline remains algorithm-agnostic.
"""

from typing import Dict, Type

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

MODEL_MAPPING: Dict[str, Type[BaseEstimator]] = {
    "LGBMRegressor": LGBMRegressor,  # type: ignore[assignment]
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
}


def get_model_class(name: str) -> Type[BaseEstimator]:
    """
    Retrieves the model class from the central registry by its string name.'

    Args:
        name (str): The user-facing name of the model.

    Returns:
        Type[BaseEstimator]: The scikit-learn compatible model class.

    Raises:
        ValueError: If the requested model name is not registered.
    """
    model_class = MODEL_MAPPING.get(name)
    if model_class is None:
        raise ValueError(
            f"Unsupported model: '{name}'. Available models are: {list(MODEL_MAPPING.keys())}"
        )
    return model_class
