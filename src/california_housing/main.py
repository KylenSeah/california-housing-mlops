"""
Application Entry Point.

This module serves as the Command Line Interface (CLI) for the California Housing
Machine Learning System. It routes user commands (train, predict) to the
appropriate pipeline orchestrators.
"""

import logging
import sys

import click
from pandera.errors import SchemaErrors
from pydantic import ValidationError

from california_housing.components.data_ingestion import DataIngestion
from california_housing.components.data_preparation import DataPreparation
from california_housing.components.model_evaluator import ModelEvaluator
from california_housing.components.model_predictor import (
    Prediction,
    PredictionOutputSchema,
)
from california_housing.components.model_registry import ModelRegistry
from california_housing.components.model_trainer import ModelTrainer
from california_housing.core.config import ConfigurationManager
from california_housing.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    IngestionError,
    ModelRegistryError,
    ModelTrainingError,
)
from california_housing.core.version import __version__
from california_housing.models.catalog import MODEL_MAPPING
from california_housing.pipeline.prediction_pipeline import PredictionPipeline
from california_housing.pipeline.training_pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config", default="config.yaml", help="Filename of the configuration file."
)
@click.option("--verbose", is_flag=True, help="Enable debug-level logging.")
@click.pass_context
def cli(ctx, config: str, verbose: bool) -> None:
    """
    California Housing MLOps CLI.

    Root command group that initializes the application context (logging, config).

    Args:
        ctx (click.Context): The click execution context object.
        config (str): Path to the configuration YAML file.
        verbose (bool): If True, sets logging level to DEBUG.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose


# =============================================================================
# TRAIN COMMAND
# =============================================================================


@cli.command()
@click.option(
    "--alias",
    default=None,
    help="Override the default MLflow registration alias defined in config.yaml.",
)
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def train(
    ctx,
    model_names: list[str],
    alias: str | None = None,
) -> None:
    """
    Train one or more models.

    Execute the end-to-end training pipeline (Ingest -> Prep -> Train -> Eval -> Register)
    for the specified models.

    Args:
        ctx (click.Context): The click execution context.
        model_names (tuple[str, ...]): List of models to train (or "all").
        alias (Optional[str]): MLflow alias tag (e.g. 'champion').
    """
    try:
        config_filename = ctx.obj["config"]
        config_manager = ConfigurationManager(config_filename=config_filename)
    except ConfigurationError as e:
        verbose = ctx.obj["verbose"]
        logger.error(f"Startup Failed: {e}", exc_info=verbose)
        sys.exit(1)
    except Exception as e:
        logger.critical(
            "CRITICAL: Unexpected System Crash during startup: %s", e, exc_info=True
        )
        sys.exit(1)

    if "all" in model_names:
        model_names = list(MODEL_MAPPING.keys())
    else:
        model_names = list(model_names)

    failed_models: list[str] = []

    for model_name in model_names:
        try:
            logger.info("=" * 40)
            logger.info("   Orchesstrating Pipeline: %s", model_name)
            logger.info("=" * 40)

            # ---------------------------------------------------------
            # DEPENDENCY INJECTION (Wiring the Components)
            # ---------------------------------------------------------
            data_ingestion = DataIngestion(
                config=config_manager.get_data_ingestion_config()
            )

            data_preparation = DataPreparation(
                config=config_manager.get_data_preparation_config(),
                globals_config=config_manager.get_globals_config(),
            )

            model_evaluator = ModelEvaluator(
                eval_config=config_manager.get_model_evaluation_config(),
                artifacts_config=config_manager.get_artifacts_config(),
            )

            model_trainer = ModelTrainer(
                config=config_manager.get_model_trainer_config(),
                globals_config=config_manager.get_globals_config(),
                data_prep_config=config_manager.get_data_preparation_config(),
                model_name=model_name,
                model_evaluator=model_evaluator,
            )

            model_registry = ModelRegistry(
                mlflow_config=config_manager.get_mlflow_config(),
                model_name=model_name,
                reg_config=config_manager.get_model_registry_config(),
            )

            pipeline = TrainingPipeline(
                model_name=model_name,
                config_manager=config_manager,
                data_ingestion=data_ingestion,
                data_preparation=data_preparation,
                model_trainer=model_trainer,
                model_evaluator=model_evaluator,
                model_registry=model_registry,
                alias=alias,  # add this
            )

            # ---------------------------------------------------------
            # EXECUTION
            # ---------------------------------------------------------
            run_id = pipeline.run()

            logger.info("--- Pipeline Success: %s (Run ID: %s) ---", model_name, run_id)

        except (
            IngestionError,
            DataValidationError,
            ModelTrainingError,
            ModelRegistryError,
            SchemaErrors,
            ValidationError,
        ) as e:
            logger.error(
                "Training failed for '%s': %s. See logs above for details.",
                model_name,
                e,
            )
            failed_models.append(model_name)
            continue

        except Exception as e:
            logger.critical(
                "--- Critical Pipeline Crash [%s] ---: %s", model_name, e, exc_info=True
            )
            failed_models.append(model_name)
            continue

    if failed_models:
        logger.error("The following models failed to train: %s", failed_models)
        sys.exit(1)

    logger.info("All requested models trained successfully.")


# =============================================================================
# PREDICT COMMAND
# =============================================================================


@cli.command()
@click.option(
    "--json", "input_json", default=None, help="JSON string for a single record."
)
@click.option(
    "--file",
    "input_file",
    default=None,
    help="Path to a CSV file for batch prediction.",
)
@click.option("--output-json", is_flag=True, help="Output predictions as JSON.")
@click.option("--model-name", default=None, help="The model architecture to use.")
@click.option(
    "--model-version", default=None, help="Specific model version (optional)."
)
@click.option(
    "--model-alias", default=None, help="Specific model alias (e.g., 'champion')."
)
@click.pass_context
def predict(
    ctx,
    input_json: str | None = None,
    input_file: str | None = None,
    output_json: bool = False,
    model_name: str | None = None,
    model_version: str | None = None,
    model_alias: str | None = None,
) -> None:
    """
    Run inference using a trained model.

    Supports both single-record JSON input and batch CSV processing.
    Outputs results to stdout (logs) or JSON (pipable).

    Args:
        ctx (click.Context): Click context.
        input_json (str): Raw JSON string input.
        input_file (str): Path to input file.
        output_json (bool): If True, suppresses logs and outputs raw JSON result.
        model_name (str): Model name override.
        model_version (str): Specific registry version.
        model_alias (str): Specific registry alias.
    """
    try:
        config_filename = ctx.obj["config"]
        config_manager = ConfigurationManager(config_filename=config_filename)
    except ConfigurationError as e:
        verbose = ctx.obj["verbose"]
        logger.error(f"Startup Failed: {e}", exc_info=verbose)
        sys.exit(1)
    except Exception as e:
        logger.critical(
            "CRITICAL: Unexpected System Crash during startup: %s", e, exc_info=True
        )
        sys.exit(1)

    # CLI-Level Validation (Fast Fail)
    if input_file and input_json:
        raise click.UsageError("Illegal Usage: Provide --file OR --json, not both.")

    if not input_file and not input_json:
        raise click.UsageError("Missing Input: Must provide --file or --json.")

    try:
        prediction_service = Prediction(config=config_manager.get_prediction_config())

        pipeline = PredictionPipeline(
            config_manager=config_manager,
            input_json=input_json,
            input_file=input_file,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            prediction_service=prediction_service,
        )
        prediction_result = pipeline.run()
        run_id = prediction_result.run_id
        results = prediction_result.output

        logger.info("--- Prediction Success (Run ID: %s) ---", run_id)

        if output_json:
            # Output pure JSON for piping to other tools (jq, etc.)
            print(results.model_dump_json(indent=4))
        else:
            # Output Human-Readable Logs
            logger.info("Service Version: %s", results.service_version)
            logger.info("Model Version: %s", results.mlflow_model_used)

            if results.predictions:
                logger.info("--- Result Preview ---")
                for i, pred in enumerate(results.predictions[:5]):
                    logger.info(f"Record {i + 1}: {pred:,.2f}")

                if len(results.predictions) > 5:
                    logger.info("and %s more...", len(results.predictions) - 5)
            else:
                logger.warning(
                    "Pipeline succeeded, but returned an empty prediction set."
                )

    except Exception as e:
        # Check if it's a "known" operational error
        is_operational = isinstance(
            e, (IngestionError, DataValidationError, ModelRegistryError, ValueError)
        )

        if is_operational:
            logger.error("Prediction Failure: %s. See logs above for details.", e)
        else:
            logger.critical("CRITICAL_FAILURE: %s", e, exc_info=True)

        if output_json:
            error_response = PredictionOutputSchema(
                errors=str(e),
                service_version=__version__,
                mlflow_model_used="Unknown",
            )
            print(error_response.model_dump_json(indent=4))
        else:
            logger.info("Service Version: %s", __version__)
            logger.info("MLflow Model: Unknown")

        sys.exit(1)


if __name__ == "__main__":
    cli()
