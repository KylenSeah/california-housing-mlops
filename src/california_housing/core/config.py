"""
Configuration Loader Module.

This module is responsible for locating, loading, and validating the configuration
YAML file. It serves as the single source of truth for the application's runtime
settings, bridging the gap between the static YAML file and the strictly typed
Pydantic schemas defined in `config_definitions.py`.
"""

import logging
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from california_housing.core.config_definitions import (
    ArtifactsConfig,
    ConfigSchema,
    DataIngestionConfig,
    DataPreparationConfig,
    GlobalsConfig,
    MlflowConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
    PredictionConfig,
    TrainerEvaluationConfig,
)
from california_housing.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def _find_config_file(config_filename: str = "config.yaml") -> Path:
    """
    Search for the configuration file in standard locations with precedence.

    Search Order:
    1. CONFIG_PATH environment variable (Highest priority).
    2. Current working directory.
    3. `configs/` directory in current working directory.
    4. Project root directory.
    5. `configs/` directory in project root.

    Args:
        config_filename (str): The name of the YAML file. Defaults to "config.yaml".

    Raises:
        FileNotFoundError: If the file cannot be found in any search path.

    Returns:
        Path: The absolute path to the resolved configuration file.
    """
    env_config_path = os.getenv("CONFIG_PATH")
    if env_config_path:
        env_path = Path(env_config_path)
        if env_path.is_file():
            logger.info(f"Loading config from env var: {env_path}")
            return env_path
        else:
            logger.warning(f"CONFIG_PATH set to '{env_path}' but file does not exist.")

    project_root = Path(__file__).resolve().parent.parent.parent.parent

    search_paths = [
        Path.cwd() / config_filename,
        Path.cwd() / "configs" / config_filename,
        project_root / config_filename,
        project_root / "configs" / config_filename,
    ]

    for path in search_paths:
        if path.is_file():
            logger.info(f"Found config file at: {path}")
            return path
    raise FileNotFoundError(
        f"Config file '{config_filename}' not found. Checked: {[str(p) for p in search_paths]}"
    )


def _load_and_validate_config(config_path: Path) -> ConfigSchema:
    """
    Read YAML and validate against strict Pydantic schemas.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        ConfigSchema: The fully validated configuration object.

    Raises:
        ConfigurationError: If YAML is malformed or schema validation fails.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    validated_config = ConfigSchema.model_validate(config_dict)
    logger.info("Configuration successfully validated.")
    return validated_config


class ConfigurationManager:
    """
    The central access point for application configuration.

    This class handles the initialization of the configuration system and provides
    facade methods (getters) to retrieve specific configuration blocks.
    """

    def __init__(self, config_filename: str = "config.yaml") -> None:
        """
        Initialize the Configuration Manager.

        Args:
            config_filename (str): Name of the config file. Defaults to "config.yaml".

        Raises:
            ConfigurationError: If the config file is missing, invalid, or violates schema.
        """
        try:
            config_path = _find_config_file(config_filename)
            self.config = _load_and_validate_config(config_path)
            logger.info("Configuration successfully loaded and validated.")
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Startup failed: Config file '{config_filename}' not found. {e}"
            ) from e
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Startup failed: Config file '{config_filename}' contains invalid YAML. {e}"
            ) from e
        except ValidationError as e:
            raise ConfigurationError(
                f"Startup failed: Config schema validation failed. {e}"
            ) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return self.config.data_ingestion

    def get_data_preparation_config(self) -> DataPreparationConfig:
        return self.config.data_preparation

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return self.config.model_trainer

    def get_model_evaluation_config(self) -> TrainerEvaluationConfig:
        return self.config.model_trainer.evaluation

    def get_model_registry_config(self) -> ModelRegistryConfig:
        return self.config.model_registry

    def get_prediction_config(self) -> PredictionConfig:
        return self.config.prediction

    def get_mlflow_config(self) -> MlflowConfig:
        return self.config.globals.mlflow

    def get_artifacts_config(self) -> ArtifactsConfig:
        return self.config.globals.artifacts

    def get_globals_config(self) -> GlobalsConfig:
        return self.config.globals

    def get_full_config(self) -> ConfigSchema:
        return self.config
