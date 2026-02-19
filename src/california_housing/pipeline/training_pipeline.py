"""
Orchestration module for the End-to-End Model Training Pipeline.

This module defines the `TrainingPipeline` class, which serves as the central
controller for the machine learning workflow. It executes the following sequential steps:
1. Data Ingestion (SQLite/Source)
2. Data Preparation (Splitting & Stratification)
3. Model Training (Preprocessing & Fitting)
4. Model Evaluation (Test Set Scoring)
5. Model Registration (MLflow Registry)

It wraps the execution in robust error handling to ensure that infrastructure,
data quality, or configuration failures are logged with specific telemetry tags
in MLflow before the process terminates.
"""

import logging
import traceback
from pathlib import Path
from typing import List

import mlflow
from mlflow import lightgbm as mlflow_lightgbm
from mlflow import sklearn as mlflow_sklearn
from mlflow import xgboost as mlflow_xgboost
from mlflow.models.signature import infer_signature
from pandera.errors import SchemaErrors
from pydantic import ValidationError

from california_housing.components.data_ingestion import DataIngestion
from california_housing.components.data_preparation import DataPreparation
from california_housing.components.model_evaluator import ModelEvaluator
from california_housing.components.model_registry import ModelRegistry
from california_housing.components.model_trainer import ModelTrainer
from california_housing.core.config import ConfigurationManager
from california_housing.core.exceptions import (
    DataValidationError,
    IngestionError,
    ModelRegistryError,
    ModelTrainingError,
    TelemetryError,
)
from california_housing.utils.model_naming import get_registered_model_name

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the complete machine learning training lifecycle.

    This class acts as a facade, coordinating the interaction between specialized
    components (Ingestion, Preparation, Trainer, Evaluator, Registry). It manages
    the MLflow run context, ensures configuration compliance, and creates a
    centralized point for catching and logging domain-specific exceptions.
    """

    def __init__(
        self,
        model_name: str,
        config_manager: ConfigurationManager,
        data_ingestion: DataIngestion,
        data_preparation: DataPreparation,
        model_trainer: ModelTrainer,
        model_evaluator: ModelEvaluator,
        model_registry: ModelRegistry,
        alias: str | None = None,
    ) -> None:
        """
        Initialize the pipeline with injected dependencies.

        Args:
            model_name (str): The unique identifier for the model architecture
                (e.g., "LGBMRegressor") used for artifact naming and registry lookups.
            config_manager (ConfigurationManager): The provider of validated configuration
                objects for all pipeline steps.
            data_ingestion (DataIngestion): Component responsible for loading raw data.
            data_preparation (DataPreparation): Component responsible for splitting
                and stratifying data.
            model_trainer (ModelTrainer): Component responsible for feature engineering
                and model fitting.
            model_evaluator (ModelEvaluator): Component responsible for scoring predictions
                and saving metrics.
            model_registry (ModelRegistry): Component responsible for promoting the
                trained model to the MLflow Model Registry.
            alias (str | None, optional): A deployment alias (e.g., "staging", "champion")
                to apply to the registered model version. Defaults to None.
        """
        self.model_name = model_name
        self.alias = alias
        self.telemetry_config = config_manager.get_telemetry_config()
        self.config_manager = config_manager

        self.ingestion = data_ingestion
        self.preparation = data_preparation
        self.trainer = model_trainer
        self.evaluator = model_evaluator
        self.registry = model_registry

    def run(self) -> str:
        """
        Executes the end-to-end training workflow.

        This method initializes an MLflow run, configures telemetry (including autolog
        for automatic logging of training errors/metrics like MSE to enable generalization
        gap analysis), and executes the pipeline steps in order. It handles all high-level
        exceptions, tagging the MLflow run with the specific failure category (Infrastructure,
        Data Quality, Model, etc.) to aid debugging.

        Raises:
            IngestionError: If the data source cannot be accessed.
            SchemaErrors: If the input data violates the Pandera schema.
            DataValidationError: If statistical checks (e.g., stratification bins) fail.
            ValidationError: If Pydantic configuration validation fails.
            ModelTrainingError: If the model fails to converge or fit.
            ModelRegistryError: If MLflow model registration fails.
            TelemetryError: If configuration logging fails.

        Returns:
            str: The unique MLflow Run ID for the completed execution.
        """
        logger.info("--- Starting training pipeline for model: %s ---", self.model_name)
        current_step = "initialization"
        temp_files: List[Path] = []

        self._setup_telemetry()

        mlflow_config = self.config_manager.get_mlflow_config()
        mlflow.set_experiment(mlflow_config.experiment_name)

        with mlflow.start_run(run_name=f"Pipeline_{self.model_name}") as active_run:
            run_id = active_run.info.run_id

            try:
                full_config = self.config_manager.get_full_config()
                mlflow.log_dict(
                    full_config.model_dump(mode="json"), "config/full_config.json"
                )
                logger.info("Configuration logged to MLflow (run_id: %s)", run_id)
            except Exception as e:
                raise TelemetryError(
                    f"Compliance Violation: Failed to log configuration. "
                    f"Run aborted to prevent creation of un-auditable artifacts. Error: {e}"
                ) from e

            mlflow.log_param("model_to_train", self.model_name)
            mlflow.set_tag("pipeline_status", "running")

            try:
                # STEP 1: Data Ingestion
                current_step = "data_ingestion"
                logger.info("Step 1: Data Ingestion")
                mlflow.set_tag("current_step", current_step)
                raw_df = self.ingestion.get_data()

                # STEP 2: Data Preparation
                current_step = "data_preparation"
                logger.info("Step 2: Data Preparation")
                mlflow.set_tag("current_step", current_step)
                splits = self.preparation.split_data(raw_df)

                # STEP 3: Model Training
                current_step = "model_training"
                logger.info("Step 3: Model Training")
                mlflow.set_tag("current_step", current_step)
                training_result = self.trainer.train(splits)

                fitted_pipeline = training_result.fitted_pipeline

                # Log Validation Metrics
                mlflow.log_metrics(training_result.validation_metrics)

                reg_model_name = get_registered_model_name(
                    mlflow_config, self.model_name
                )

                logger.info(
                    "Logging model artifacts to MLflow as '%s'...", reg_model_name
                )

                signature_sample = splits.X_test.iloc[:100]
                signature = infer_signature(
                    model_input=signature_sample,
                    model_output=fitted_pipeline.predict(signature_sample),
                )

                # Log Trained Model
                mlflow_sklearn.log_model(
                    sk_model=fitted_pipeline,
                    name="model",
                    signature=signature,
                    input_example=splits.X_test.iloc[:5],
                    registered_model_name=reg_model_name,
                )

                # STEP 4: Model Evaluation
                current_step = "model_evaluation"
                logger.info("Step 4: Model Evaluation")
                mlflow.set_tag("current_step", current_step)

                metrics = self.evaluator.evaluate(
                    pipeline=fitted_pipeline,
                    X=splits.X_test,
                    y=splits.y_test,
                    set_name="test",
                )

                # Log Test Metrics
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

                metrics_path = self.evaluator.save_metrics(
                    metrics=metrics,
                    model_name=self.model_name,
                    set_name="test",
                    run_id=run_id,
                )

                temp_files.append(metrics_path)

                mlflow.log_artifact(str(metrics_path))

                # STEP 5: Registry
                registry_config = self.config_manager.get_model_registry_config()
                if registry_config.enabled:
                    current_step = "model_registry"
                    logger.info("Step 5: Model Registration")
                    mlflow.set_tag("current_step", current_step)
                    self.registry.register_latest_version(
                        run_id=run_id,
                        alias=self.alias,
                    )

                mlflow.set_tag("pipeline_status", "completed")
                mlflow.set_tag("current_step", "completed")

                return str(run_id)

            # --- Centralized Error Handling ---

            # 1. Infrastructure Failures (DB down, File missing) (DataIngestion)
            except IngestionError as e:
                self._handle_ingestion_error(e, current_step)
                raise

            # 2a. Data Structure Failures (Pandera: Wrong types, nulls) (DataIngestion)
            except SchemaErrors as e:
                self._handle_data_quality_error(e, current_step)
                raise

            # 2b. Data Statistical Failures (Custom: Insufficient bins/samples) (DataPreparation)
            except DataValidationError as e:
                self._handle_statistical_error(e, current_step)
                raise

            # 3. DYNAMIC CONFIGURATION FAILURES (Pydantic)
            except ValidationError as e:
                self._handle_config_error(e, current_step)
                raise

            # 4. Training Logic Failures (Convergence issues, NaN gradients, CV splits) (ModelTrainer)
            except ModelTrainingError as e:
                self._handle_training_error(e, current_step)
                raise

            # 5. Deployment Failures (MLflow server outage, Permissions, Naming conflicts) (ModelRegistry)
            except ModelRegistryError as e:
                self._handle_registry_error(e, current_step)
                raise

            # 6. Unanticipated System Crashes (Bugs, OOM, Library changes)
            except Exception as e:
                self._handle_unknown_error(e, current_step)
                raise

            finally:
                self._cleanup_temp_files(temp_files)

    # --- Pipeline Lifecycle Helpers ---

    def _setup_telemetry(self) -> None:
        """Configures MLflow autolog based on telemetry config."""
        if not self.telemetry_config.autolog_enabled:
            logger.info("Autologging disabled via config.")
            return

        try:
            mlflow_sklearn.autolog(
                log_models=self.telemetry_config.autolog_log_models,
                silent=self.telemetry_config.autolog_silent,
            )
            mlflow_lightgbm.autolog(
                log_models=self.telemetry_config.autolog_log_models,
                silent=self.telemetry_config.autolog_silent,
            )
            mlflow_xgboost.autolog(
                log_models=self.telemetry_config.autolog_log_models,
                silent=self.telemetry_config.autolog_silent,
            )
            logger.info("MLflow autologging configured.")
        except Exception as e:
            logger.warning(f"Telemetry setup failed (continuing without autolog): {e}")

    def _cleanup_temp_files(self, temp_files: List[Path]) -> None:
        """
        Deletes temporary files created during the pipeline run.

        Args:
            temp_files (List[Path]): A list of file paths to remove.
        """
        if not temp_files:
            return

        logger.info("Cleaning up %s temporary files...", len(temp_files))
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug("Deleted temp file: %s", temp_file)
            except Exception as cleanup_e:
                logger.warning("Failed to delete %s: %s", temp_file, cleanup_e)

    # --- Specialized Error Handlers ---

    def _handle_ingestion_error(self, e: IngestionError, step: str) -> None:
        """
        Handles infrastructure-related failures (DB connectivity, file access).

        Logs the error and updates MLflow tags to indicate an 'infrastructure_error'.

        Args:
            e (IngestionError): The exception raised during ingestion.
            step (str): The pipeline step name where the error occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Infrastructure failure at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "infrastructure_error")
            mlflow.set_tag("failure_step", step)

            mlflow.log_text(
                f"Error: {e}\n\nACTION: Check database availability, permissions and file paths.",
                "infrastructure_error.txt",
            )

    def _handle_data_quality_error(self, e: SchemaErrors, step: str) -> None:
        """
        Handles Pandera schema validation failures.

        Logs the dataframe of failure cases to MLflow for debugging and tags the
        failure as a 'schema_violation'.

        Args:
            e (SchemaErrors): The Pandera exception containing failure cases.
            step (str): The pipeline step where validation failed.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Data Quality check failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "data_quality_error")
            # We add failure detail tag as this failue category holds 2 failures.
            # To differentiate at a glance in the UI without going into the log_text.
            mlflow.set_tag("failure_detail", "schema_violation")
            mlflow.set_tag("failure_step", step)

            try:
                violations_csv = e.failure_cases.to_csv(index=False)
                mlflow.log_text(violations_csv, "data_validation_failures.csv")
            except Exception:
                mlflow.log_text(str(e), "data_validation_error.txt")

    def _handle_statistical_error(self, e: DataValidationError, step: str) -> None:
        """
        Handles statistical or logical data validation failures (e.g., empty tables, bad stratification).

        Args:
            e (DataValidationError): The custom validation exception.
            step (str): The pipeline step where validation failed.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Data Validation failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "data_quality_error")
            mlflow.set_tag("failure_detail", "statistical_validation")
            mlflow.set_tag("failure_step", step)

            mlflow.log_text(str(e), "data_validation_error.txt")

    def _handle_config_error(self, e: ValidationError, step: str) -> None:
        """
        Handles Pydantic configuration validation failures.

        Args:
            e (ValidationError): The Pydantic exception.
            step (str): The pipeline step where configuration was checked.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Configuration invalid at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "configuration_error")
            mlflow.set_tag("failure_step", step)

            mlflow.log_text(str(e), "config_error.txt")

    def _handle_training_error(self, e: ModelTrainingError, step: str) -> None:
        """
        Handles failures during the model fitting or internal validation process.

        Args:
            e (ModelTrainingError): The training exception.
            step (str): The pipeline step where training failed.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Model Training failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "model_error")
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(str(e), "training_error.txt")

    def _handle_registry_error(self, e: ModelRegistryError, step: str) -> None:
        """
        Handles failures when communicating with the Model Registry.

        Args:
            e (ModelRegistryError): The registry exception.
            step (str): The pipeline step where registration failed.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "Model Registration failed at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "deployment _error")
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(
                f"Registry Error: {e}\n\nACTION: Check MLflow server status, credentials, "
                "and artifact storage latency.",
                "registry_error.txt",
            )

    def _handle_unknown_error(self, e: Exception, step: str) -> None:
        """
        Handles unexpected/uncaught exceptions (Pokemon handling: 'Catch 'em all').

        Logs the full traceback to MLflow to assist in post-mortem debugging.

        Args:
            e (Exception): The unexpected exception.
            step (str): The pipeline step where the crash occurred.
        """
        is_debug = logger.isEnabledFor(level=logging.DEBUG)
        logger.error(
            "CRITICAL: Unexpected crash at step '%s': %s", step, e, exc_info=is_debug
        )

        if mlflow.active_run():
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("failure_category", "uncaught_exception")
            mlflow.set_tag("error_type", type(e).__name__)
            mlflow.set_tag("failure_step", step)
            mlflow.log_text(traceback.format_exc(), "crash_dump.txt")
