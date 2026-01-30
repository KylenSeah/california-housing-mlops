"""
Inference component for generating predictions.

This module provides the schema definitions for API inputs/outputs and the
Prediction service. It handles the efficient loading of MLflow models
(caching them in memory) and the execution of the inference pipeline.
"""

import logging
import threading

import numpy as np
import pandas as pd
from mlflow import sklearn as mlflow_sklearn
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from california_housing.core.config_definitions import PredictionConfig
from california_housing.core.exceptions import ModelRegistryError
from california_housing.core.version import __version__
from california_housing.utils.model_naming import get_registered_model_name

logger = logging.getLogger(__name__)

# =============================================================================
# I/O SCHEMAS (The API Contract)
# =============================================================================


class HousingDataInputSchema(BaseModel):
    """
    Schema representing a single record of housing data for inference.

    This ensures that clients (CLI, API, UI) provide all necessary fields
    with the correct data types before the model is invoked.
    """

    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


class PredictionOutputSchema(BaseModel):
    """
    Standardized response format for prediction results.
    """

    predictions: list[float] | None = None
    errors: str | None = None
    service_version: str
    mlflow_model_used: str | None = None


# =============================================================================
# MODEL LOADER (Singleton with Caching)
# =============================================================================


class ModelLoader:
    """
    Thread-safe model loader with in-memory caching.

    This class ensures that the heavy model artifact is loaded from disk/MLflow
    only once, rather than reloading it for every single prediction request.
    """

    _model_pipeline: Pipeline | None = None
    _loaded_model_identifier: str | None = None
    _lock = threading.Lock()

    @classmethod
    def get_model(
        cls,
        mlflow_config,
        model_name: str,
        model_version: str | None = None,
        model_alias: str | None = None,
    ) -> Pipeline:
        """
        Retrieve the Scikit-Learn pipeline from the MLflow registry.

        Args:
            mlflow_config (MlflowConfig): Config with registry prefix.
            model_name (str): Algorithm name (e.g. 'LGBMRegressor').
            model_version (str, optional): Specific version number.
            model_alias (str, optional): Specific alias (e.g. 'champion').
                Defaults to 'staging' if neither version nor alias is provided.

        Returns:
            Pipeline: The loaded Scikit-Learn pipeline.

        Raises:
            ModelRegistryError: If the model cannot be downloaded or loaded.
        """

        # Determine the URI based on priority: Alias > Version > Default(staging)
        reg_name = get_registered_model_name(mlflow_config, model_name)
        if model_alias:
            model_identifier = f"models:/{reg_name}@{model_alias}"
        elif model_version:
            model_identifier = f"models:/{reg_name}/{model_version}"
        else:
            # Default to staging for safety
            model_identifier = f"models:/{reg_name}@staging"
            logger.info("Defaulting to model identifier: %s", model_identifier)

        # Double-Check Locking Pattern for Thread Safety
        with cls._lock:
            # If we already have this exact model loaded, return it (Cache Hit)
            if (
                cls._model_pipeline is not None
                and cls._loaded_model_identifier == model_identifier
            ):
                logger.debug("Model cached: %s", model_identifier)
                return cls._model_pipeline

            # Cache Miss - Load from MLflow
            logger.info("Loading model from registry: %s", model_identifier)
            try:
                # mlflow.sklearn.load_model returns the native sklearn object
                cls._model_pipeline = mlflow_sklearn.load_model(model_identifier)
                cls._loaded_model_identifier = model_identifier
                logger.info("Model loaded successfully.")

            except Exception as e:
                # Reset cache on failure to prevent stale state
                cls._model_pipeline = None
                cls._loaded_model_identifier = None
                raise ModelRegistryError(
                    f"Failed to load model {model_identifier}: {e}"
                ) from e

        assert cls._model_pipeline is not None, (
            "Critical Logic Error: Model is None after successful load."
        )

        return cls._model_pipeline


# =============================================================================
# PREDICTION SERVICE
# =============================================================================


class Prediction:
    """
    Facade for executing predictions.

    This class orchestrates the data conversion (Pydantic -> DataFrame) and
    delegates the actual inference to the ModelLoader.
    """

    def __init__(self, config: PredictionConfig):
        self.config = config

    def make_predictions(
        self,
        input_data: list[HousingDataInputSchema],
        model_name: str | None = None,
        model_version: str | None = None,
        model_alias: str | None = None,
    ) -> PredictionOutputSchema:
        """
        Execute the inference logic on a batch of validated inputs.

        Args:
            input_data (list[HousingDataInputSchema]): Validated input records.
            model_name (str, optional): Override default model name.
            model_version (str, optional): Specific version to use.
            model_alias (str, optional): Specific alias to use.

        Returns:
            PredictionOutputSchema: The predictions and metadata.

        Raises:
            ValueError: If the model produces NaNs or Infs.
        """
        active_model = model_name or self.config.default_model_name

        # 1. Convert Input Schema -> Pandas DataFrame
        input_df = pd.DataFrame([record.model_dump() for record in input_data])

        # 2. Load Model (Cached)
        pipeline = ModelLoader.get_model(
            mlflow_config=self.config.mlflow,
            model_name=active_model,
            model_version=model_version,
            model_alias=model_alias,
        )

        # 3. Inference
        raw_preds = pipeline.predict(input_df)

        # 4. Safety Checks
        if np.isnan(raw_preds).any() or np.isinf(raw_preds).any():
            raise ValueError("Model produced NaNs or Infs in predictions.")

        # 5. Format Output
        predictions = [float(p) for p in raw_preds]

        return PredictionOutputSchema(
            predictions=predictions,
            service_version=__version__,
            mlflow_model_used=ModelLoader._loaded_model_identifier,
        )
