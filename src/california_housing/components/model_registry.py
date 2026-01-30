"""
Model Registry interaction component.

This module handles the promotion of trained models within the MLflow Model
Registry. It abstracts the complexity of asynchronous model registration
and aliasing (e.g., promoting a model to '@staging').
"""

import logging
import time

from mlflow import MlflowClient, MlflowException

from california_housing.core.config_definitions import MlflowConfig, ModelRegistryConfig
from california_housing.core.exceptions import ModelRegistryError
from california_housing.utils.model_naming import get_registered_model_name

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model versioning and aliasing in MLflow.

    This class ensures that a successfully trained model run is correctly
    identified and tagged in the central registry, making it available for
    downstream deployment.
    """

    def __init__(
        self,
        mlflow_config: MlflowConfig,
        reg_config: ModelRegistryConfig,
        model_name: str,
        client: MlflowClient | None = None,
    ) -> None:
        """
        Initialize the registry handler.

        Args:
            mlflow_config (MlflowConfig): Global MLflow settings.
            reg_config (ModelRegistryConfig): Registry-specific settings (enabled/alias).
            model_name (str): The algorithm name (e.g., 'LGBMRegressor').
            client (Optional[MlflowClient]): Injectable client for testing.
        """
        self.mlflow_config = mlflow_config
        self.model_name = model_name
        self.reg_config = reg_config

        # Use injected client (for tests) or create a new one
        self.client = client or MlflowClient()

    def register_latest_version(
        self,
        run_id: str,
        alias: str | None = None,
        max_retries: int = 5,
    ) -> None:
        """
        Locate the model artifact for a specific run and apply a registry alias.

        This method includes a polling mechanism (retry loop) because MLflow
        model creation is asynchronous. It waits for the model version to
        become available before attempting to alias it.

        Args:
            run_id (str): The MLflow Run ID containing the trained model.
            alias (Optional[str]): The alias to apply (e.g., 'champion').
                If None, uses the target_alias from config.
            max_retries (int): Number of times to poll for the model version.

        Raises:
            ModelRegistryError: If the model cannot be found or aliasing fails.
        """
        target_alias = alias or self.reg_config.target_alias

        registered_model_name = get_registered_model_name(
            self.mlflow_config, self.model_name
        )

        logger.info(
            "Attempting to alias model '%s' (Run ID: %s) as '@%s'...",
            registered_model_name,
            run_id,
            target_alias,
        )

        for attempt in range(max_retries):
            try:
                # 1. Search for the specific version linked to this Run ID
                latest_versions = self.client.search_model_versions(
                    filter_string=f"name='{registered_model_name}' AND run_id='{run_id}'",
                    order_by=["version_number DESC"],
                    max_results=1,
                )

                if latest_versions:
                    target_version = latest_versions[0]

                    # 2. Apply the Alias (Promotion)
                    self.client.set_registered_model_alias(
                        name=registered_model_name,
                        alias=target_alias,
                        version=target_version.version,
                    )
                    logger.info(
                        "SUCCESS: Aliased '%s' (v%s) as '@%s' (Run ID: %s)",
                        registered_model_name,
                        target_version.version,
                        target_alias,
                        run_id,
                    )

                    return

                # 3. Backoff Strategy (Wait if not found yet)
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s...
                    logger.warning(
                        "Model '%s' not found yet (Run ID: %s). "
                        "Attempt %s/%s failed. Retrying in %ss...",
                        registered_model_name,
                        run_id,
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)

            except MlflowException as e:
                # Handle cases where the Registered Model itself might not exist yet
                if "RESOURCE_DOES_NOT_EXIST" in str(e) and attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Model lookup failed (404 Not Found) for '%s' (Run ID: %s). "
                        "Attempt %s/%s failed. Retrying in %ss...",
                        registered_model_name,
                        run_id,
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                raise ModelRegistryError(
                    f"Registry operation failed for '{registered_model_name}' "
                    f"(Run ID: {run_id}): {e}"
                ) from e

        # If loop finishes without return, we failed.
        raise ModelRegistryError(
            f"Model '{registered_model_name}' (Run ID: {run_id}) not found "
            f"after {max_retries} attempts. Check if training step logged the model correctly."
        )
