"""
Prediction Pipeline Module

This module implements the prediction pipeline for the California Housing MLOps project.
It handles data ingestion, validation, model inference, and logging of results using MLflow.
The pipeline integrates with a prediction service to make inferences on housing data,
ensuring robust error handling and telemetry for monitoring.
"""

import json
import logging
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from pydantic import TypeAdapter, ValidationError

from california_housing.components.model_predictor import (
    HousingDataInputSchema,
    Prediction,
    PredictionOutputSchema,
)
from california_housing.core.config import ConfigurationManager
from california_housing.core.exceptions import (
    DataValidationError,
    IngestionError,
    ModelRegistryError,
    TelemetryError,
)
from california_housing.core.version import __version__

logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class PredictionPipelineResult:
    """
    Data class representing the result of the prediction pipeline execution.

    Attributes:
        run_id (str): The MLflow run ID associated with this pipeline execution.
        output (PredictionOutputSchema): The output schema containing predictions and metadata.
    """

    run_id: str
    output: PredictionOutputSchema


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================


class PredictionPipeline:
    """
    Prediction Pipeline Class

    This class orchestrates the end-to-end prediction process for housing data.
    It loads input data, validates it against a schema, performs inference using a specified model,
    and logs results and metrics to MLflow for traceability and monitoring.
    """

    def __init__(
        self,
        config_manager: ConfigurationManager,
        prediction_service: Prediction,  # added
        input_json: str | None = None,
        input_file: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        """
        Initializes the PredictionPipeline instance.

        Args:
            config_manager (ConfigurationManager): Manager for loading configuration settings.
            prediction_service (Prediction): Service responsible for making model predictions.
            input_json (str | None, optional): JSON string containing input data. Defaults to None.
            input_file (str | None, optional): Path to a file (CSV or JSON) containing input data. Defaults to None.
            model_name (str | None, optional): Name of the model to use for prediction. Defaults to None.
            model_version (str | None, optional): Version of the model to use. Defaults to None.
            model_alias (str | None, optional): Alias of the model to use (e.g., 'champion'). Defaults to None.

        Raises:
            ValueError: If both input_json and input_file are provided.
            ValueError: If neither input_json nor input_file is provided.
        """
        if input_json and input_file:
            raise ValueError("Provide input_json OR input_file, not both.")
        if not input_json and not input_file:
            raise ValueError("Must provide input data via input_json or input_file.")
        self.config_manager = config_manager
        self.input_json = input_json
        self.input_file = input_file
        self.model_name = model_name
        self.model_version = model_version
        self.model_alias = model_alias
        self.prediction_service = prediction_service

    def run(self) -> PredictionPipelineResult:
        """
        Executes the prediction pipeline.

        This method performs the following steps:
        1. Initializes an MLflow run.
        2. Loads input data.
        3. Validates the data against the input schema.
        4. Makes predictions using the prediction service.
        5. Logs results, metrics, and artifacts to MLflow.

        Returns:
            PredictionPipelineResult: The result containing the MLflow run ID and prediction output.

        Raises:
            IngestionError: If data loading fails.
            DataValidationError: If data validation fails.
            ModelRegistryError: If model loading from registry fails.
            ValueError: If model inference encounters a value-related issue.
            Exception: For any unhandled errors during execution.
        """
        start_time = time.time()
        current_step = "initialization"

        default_model = self.config_manager.get_prediction_config().default_model_name
        resolved_model = self.model_name or default_model

        mlflow_config = self.config_manager.get_mlflow_config()
        mlflow.set_experiment(mlflow_config.experiment_name)

        run_name = f"Prediction_{resolved_model}_{int(start_time)}"

        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id
            mlflow.set_tag("pipeline_status", "running")
            mlflow.set_tag("current_step", current_step)

            mlflow.log_params(
                {
                    "input_type": "file" if self.input_file else "json",
                    "requested_model_name": self.model_name or "none",
                    "requested_model_version": self.model_version or "none",
                    "requested_model_alias": self.model_alias or "none",
                    "resolved_model_name": resolved_model,
                    "service_version": __version__,
                }
            )

            try:
                # Step 1: Ingestion
                current_step = "data_ingestion"
                logger.info("Step 1: Loading input data from source...")
                mlflow.set_tag("current_step", current_step)
                df = self._load_data()

                # Step 2: Validation
                current_step = "data_validation"
                logger.info("Step 2: Validating %s records against schema...", len(df))
                mlflow.set_tag("current_step", current_step)
                validated_inputs = self._validate_data(df)

                # Step 3: Prediction
                current_step = "prediction"
                logger.info("Step 3: Executing model inference...")
                mlflow.set_tag("current_step", current_step)
                results = self.prediction_service.make_predictions(
                    input_data=validated_inputs,
                    model_name=resolved_model,
                    model_version=self.model_version,
                    model_alias=self.model_alias,
                )

                # Step 4: Logging
                current_step = "logging"
                logger.info("Step 4: Persisting telemetry and artifacts...")
                mlflow.set_tag("current_step", current_step)
                try:
                    self._log_results(df, results, start_time)
                except TelemetryError as e:
                    self._handle_telemetry_error(e, current_step)

                logger.info("Pipeline execution successful.")
                mlflow.set_tag("pipeline_status", "completed")
                mlflow.set_tag("current_step", "completed")

                return PredictionPipelineResult(run_id=str(run_id), output=results)

            except IngestionError as e:
                self._handle_ingestion_error(e, current_step)
                raise

            except DataValidationError as e:
                self._handle_data_quality_error(e, current_step)
                raise

            except ModelRegistryError as e:
                self._handle_registry_error(e, current_step)
                raise

            except ValueError as e:
                self._handle_model_error(e, current_step)
                raise

            except Exception as e:
                self._handle_unknown_error(e, current_step)
                raise

    # =============================================================================
    # INTERNAL HELPERS (Data Logic)
    # =============================================================================

    def _load_data(self) -> pd.DataFrame:
        """
        Loads input data from file or JSON string.

        Supports CSV and JSON file formats. If input_json is provided, it is parsed directly.

        Returns:
            pd.DataFrame: The loaded input data as a DataFrame.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the file format is unsupported.
            ValueError: If no valid input source is provided.
            IngestionError: For any other ingestion failures.
        """
        try:
            if self.input_file:
                input_path = Path(self.input_file)
                if not input_path.is_file():
                    raise FileNotFoundError(
                        f"Input file not found at path: '{input_path}'"
                    )

                if input_path.suffix.lower() == ".json":
                    return pd.read_json(input_path)
                elif input_path.suffix.lower() == ".csv":
                    return pd.read_csv(input_path)
                else:
                    raise ValueError(
                        f"Unsupported file format: {input_path.suffix}. Use .csv or .json."
                    )

            elif self.input_json:
                data = json.loads(self.input_json)
                return pd.DataFrame(data if isinstance(data, list) else [data])
            raise ValueError(
                "Critical Logic Error: No valid input source (file or JSON) provided."
            )

        except Exception as e:
            raise IngestionError(f"Data Ingestion Failed: {e}") from e

    def _validate_data(self, df: pd.DataFrame) -> list[HousingDataInputSchema]:
        """
        Validates the input DataFrame against the HousingDataInputSchema.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            list[HousingDataInputSchema]: List of validated input schemas.

        Raises:
            DataValidationError: If the DataFrame is empty or validation fails.
        """
        if df.empty:
            raise DataValidationError("Input DataFrame is empty.")
        try:
            adapter = TypeAdapter(list[HousingDataInputSchema])
            return adapter.validate_python(df.to_dict(orient="records"))
        except ValidationError as e:
            raise DataValidationError(f"Input validation failed: {e}") from e

    def _log_results(
        self, input_df: pd.DataFrame, results: PredictionOutputSchema, start_time: float
    ) -> None:
        """
        Logs prediction results, metrics, and artifacts to MLflow.

        Calculates latency metrics and logs a Parquet file of results, along with a sample preview.

        Args:
            input_df (pd.DataFrame): The original input DataFrame.
            results (PredictionOutputSchema): The prediction output.
            start_time (float): The start time of the pipeline for latency calculation.

        Raises:
            TelemetryError: If logging to MLflow fails.
        """
        try:
            duration = time.time() - start_time
            batch_size = len(input_df)
            per_record_latency = duration / batch_size if batch_size > 0 else 0
            mlflow.log_metric("latency_per_record_s", per_record_latency)
            mlflow.log_metric("batch_size", batch_size)
            mlflow.log_metric("latency_total_s", duration)

            logger.info(
                "Batch Performance: %s rows in %.2fs (%.4fs/row)",
                batch_size,
                duration,
                per_record_latency,
            )

            log_df = input_df.copy()
            log_df["predicted_value"] = results.predictions
            log_df["model_version"] = results.mlflow_model_used
            log_df["service_version"] = results.service_version
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "prediction_payload.parquet"
                log_df.to_parquet(temp_path, index=False)
                mlflow.log_artifact(str(temp_path))
            sample_df = log_df.sample(n=min(batch_size, 100))
            mlflow.log_table(data=sample_df, artifact_file="preview_sample.json")
            logger.info("Logged prediction artifacts to MLflow.")
        except Exception as e:
            raise TelemetryError(f"Telemetry persistence failed: {e}") from e

    # =============================================================================
    # ERROR HANDLING METHODS
    # =============================================================================

    def _handle_ingestion_error(self, e: IngestionError, step: str) -> None:
        """
        Handles ingestion errors by logging details to MLflow and console.

        Args:
            e (IngestionError): The exception raised.
            step (str): The current pipeline step where the error occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Input Ingestion failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "infrastructure_error")
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(
                f"Error: {e}\n\nACTION: Check file permissions, S3 connectivity, or JSON format.",
                "ingestion_error.txt",
            )

    def _handle_data_quality_error(self, e: DataValidationError, step: str) -> None:
        """
        Handles data validation errors by logging details to MLflow and console.

        Args:
            e (DataValidationError): The exception raised.
            step (str): The current pipeline step where the error occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Data Validation rejected input at step '%s': %s",
            step,
            e,
            exc_info=is_debug,
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "client_error")
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(
                f"Validation Failed: {e}\n\nACTION: Verify input JSON against schema. Ensure no nulls in required columns.",
                "validation_error.txt",
            )

    def _handle_registry_error(self, e: ModelRegistryError, step: str) -> None:
        """
        Handles model registry errors by logging details to MLflow and console.

        Args:
            e (ModelRegistryError): The exception raised.
            step (str): The current pipeline step where the error occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Model Loading failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "model_registry_error")
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(
                f"Registry Error: {e}\n\nACTION: Check MLflow Tracking Server status and Model Name/Version.",
                "registry_error.txt",
            )

    def _handle_model_error(self, e: ValueError, step: str) -> None:
        """
        Handles model inference errors by logging details to MLflow and console.

        Args:
            e (ValueError): The exception raised.
            step (str): The current pipeline step where the error occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Model Inference failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "model_error")
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(
                f"Model Error: {e}\n\n"
                "ACTION:\n"
                "1. Check input feature shape and value ranges (outliers/zeros).\n"
                "2. Check Model Artifact integrity (e.g., StandardScaler variance=0 "
                "due to constant training features).\n"
                "3. Verify Library Version compatibility.",
                "model_error.txt",
            )

    def _handle_telemetry_error(self, e: TelemetryError, step: str) -> None:
        """
        Handles telemetry logging errors by logging a warning (pipeline continues).

        Args:
            e (TelemetryError): The exception raised.
            step (str): The current pipeline step where the error occurred.
        """
        logger.warning(
            "Telemetry failed at step '%s' (Pipeline Continuing): %s", step, e
        )

    def _handle_unknown_error(self, e: Exception, step: str) -> None:
        """
        Handles uncaught exceptions by logging crash details to MLflow and console.

        Args:
            e (Exception): The exception raised.
            step (str): The current pipeline step where the error occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.critical(
            "UNEXPECTED SYSTEM CRASH at step: '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "uncaught_exception")
            mlflow.set_tag("failure_detail", type(e).__name__)
            mlflow.set_tag("failure_step", step)

            mlflow.log_text(traceback.format_exc(), "crash_dump.txt")
