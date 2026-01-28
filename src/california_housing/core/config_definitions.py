"""
Pydantic model definitions for the pipeline configuration.

This module provides strict type-validation and 'fail-fast' checks for the
YAML configuration, ensuring the system is correctly parameterized before
any compute-intensive tasks begin.
"""

import re
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

ALLOWED_METRICS = {"rmse", "mae", "r2", "mse"}


# =============================================================================
# 1. CONSTANTS & SHARED LEAVES (Foundations)
# =============================================================================


class ArtifactsConfig(BaseModel):
    """Configuration for storing pipeline artifacts."""

    model_config = ConfigDict(extra="forbid")

    root_dir: Path = Field(description="Root directory for all pipeline artifacts.")
    model_dir: str = Field(description="Subdirectory for storing trained model files.")
    metrics_dir: str = Field(description="Subdirectory for storing evaluation metrics.")


class MlflowConfig(BaseModel):
    """Configuration for MLflow tracking and model registration."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = Field(description="Name of the MLflow experiment.")
    registered_model_prefix: str = Field(
        description="Prefix for the registered model name in the registry."
    )


# =============================================================================
# 2. HELPER & STRATEGY SCHEMAS (Branches)
# =============================================================================


class DataPrepPreprocessingConfig(BaseModel):
    """Configuration for initial data preparation steps like stratified splitting."""

    model_config = ConfigDict(extra="forbid")

    new_stratify_col: None | str = None
    stratify_on_col: None | str = None
    stratify_bins: list[float]
    stratify_labels: list[int]
    min_stratify_samples: int = Field(
        default=3, description="Min samples per bin to allow split."
    )

    @model_validator(mode="after")
    def check_bins_and_labels_length(self) -> "DataPrepPreprocessingConfig":
        """
        Verify that the number of labels matches the bin intervals.

        Raises:
            ValueError: If the labels count is not exactly one less than the bin count.
        """
        if len(self.stratify_labels) != len(self.stratify_bins) - 1:
            raise ValueError(
                "Length of stratify_labels must be one less than stratify_bins."
            )
        return self


class DataSplitConfig(BaseModel):
    """Configuration for the train-test split."""

    model_config = ConfigDict(extra="forbid")
    test_size: float = Field(
        gt=0,
        lt=1,
        description="Proportion of the dataset to include in the test split.",
    )


class FinalCleanupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle_inf: bool = Field(
        default=True, description="If True, converts np.inf/-np.inf to np.nan."
    )

    perform_imputation: bool = Field(
        default=True, description="If True, applies SimpleImputer to fill NaNs."
    )

    imputer_strategy: Literal["mean", "median", "most_frequent", "constant"] = Field(
        default="median", description="Strategy for the final imputer."
    )


class PipelinePreprocessingConfig(BaseModel):
    """Configuration for the preprocessing steps inside the model pipeline."""

    model_config = ConfigDict(extra="forbid")
    ratio_cols: dict[str, tuple[str, str]]
    log_cols: list[str]
    geo_cols: list[str]
    n_clusters: int = Field(gt=1)
    gamma: float
    imputer_strategy_num: Literal["mean", "median", "most_frequent"]
    imputer_strategy_cat: Literal["most_frequent", "constant"]
    onehot_handle_unknown: Literal["error", "ignore", "infrequent_if_exist"]
    final_cleanup: FinalCleanupConfig = Field(default_factory=FinalCleanupConfig)


class TrainerEvaluationConfig(BaseModel):
    """Configuration for how evaluation metrics are saved."""

    model_config = ConfigDict(extra="forbid")
    metrics: list[str]
    metrics_filename: str

    @field_validator("metrics")
    def check_metrics_are_supported(cls, v: list[str]) -> list[str]:
        """
        Validate that all requested metrics are implemented in the system.

        Args:
            v (List[str]): The list of metric names provided in the YAML.

        Raises:
            ValueError: If an unsupported metric string is found.
        """
        valid_set = ALLOWED_METRICS
        unsupported = set(v) - valid_set
        if unsupported:
            raise ValueError(
                f"Configuration Error: Unsupported metrics found: {unsupported}. Supported metrics are: {valid_set}"
            )
        return v


# --- Model Parameter Schemas ---
class LGBMParams(BaseModel):
    """Schema for LightGBM hyperparameter validation."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["LGBMRegressor"]
    colsample_bytree: float
    learning_rate: float
    max_depth: int
    n_estimators: int
    num_leaves: int
    objective: str
    subsample: float
    verbose: int = -1


class RandomForestParams(BaseModel):
    """Schema for Random Forest hyperparameter validation."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["RandomForestRegressor"]
    n_estimators: int
    min_samples_leaf: int
    max_features: str | float | int
    max_depth: int | None
    criterion: str


class XGBoostParams(BaseModel):
    """Schema for XGBoost hyperparameter validation."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["XGBRegressor"]
    colsample_bytree: float
    gamma: float
    learning_rate: float
    max_depth: int
    n_estimators: int
    objective: str
    subsample: float


# --- Validation Strategy Schemas ---
class SingleSplitConfig(BaseModel):
    """Settings for a simple train-validation holdout split."""

    model_config = ConfigDict(extra="forbid")
    val_size: float = Field(
        gt=0, lt=1, description="Proportion of the training set to use for validation."
    )


class CrossValidationConfig(BaseModel):
    """Settings for K-Fold cross-validation strategy."""

    model_config = ConfigDict(extra="forbid")
    n_folds: int = Field(
        gt=1, description="Number of folds to use for cross-validation."
    )


# =============================================================================
# 3. DOMAIN CONFIGURATIONS (Trunks)
# =============================================================================
class GlobalsConfig(BaseModel):
    """Global configuration settings applicable to the entire pipeline."""

    model_config = ConfigDict(extra="forbid")

    target_col: str = Field(description="The name of the target column for prediction.")
    random_state: int = Field(description="Global random state for reproducibility.")

    artifacts: ArtifactsConfig
    mlflow: MlflowConfig


class DataIngestionConfig(BaseModel):
    """Configuration for the data source from which to ingest data."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sqlite"]
    db_path: Path
    table_name: str

    @field_validator("table_name")
    def table_name_must_be_sane(cls, v: str) -> str:
        """
        Ensure the SQL table name contains only safe, alphanumeric characters.

        Args:
            v (str): The raw table name string from the configuration.

        Raises:
            ValueError: If the table name contains spaces, special
                characters, or potential SQL injection patterns.
        """
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("table_name contains invalid characters")
        return v


class DataPreparationConfig(BaseModel):
    """Container for all data preparation configurations."""

    model_config = ConfigDict(extra="forbid")
    preprocessing: DataPrepPreprocessingConfig
    split: DataSplitConfig


class ModelTrainerConfig(BaseModel):
    """Configuration for the entire model training stage."""

    model_config = ConfigDict(extra="forbid")

    validation_strategy: Literal["single_split", "cross_validation"]

    single_split_config: None | SingleSplitConfig = None
    cross_validation_config: None | CrossValidationConfig = None

    evaluation: TrainerEvaluationConfig
    preprocessing: PipelinePreprocessingConfig

    params: dict[
        str,
        Annotated[
            Union[LGBMParams, RandomForestParams, XGBoostParams],
            Field(discriminator="type"),
        ],
    ]

    quality_threshold: float = Field(
        description="Maximum allowed error (e.g. RMSE) on the validation set."
    )

    @model_validator(mode="after")
    def check_validation_configs(self) -> "ModelTrainerConfig":
        """
        Verify that the specific configuration block exists for the chosen strategy.

        Raises:
            ValueError: If the required sub-configuration (single_split or
                cross_validation) is missing for the active strategy.
        """
        strategy = self.validation_strategy
        if strategy == "single_split":
            if self.single_split_config is None:
                raise ValueError(
                    "`single_split_config` must be provided for 'single_split' strategy."
                )
        elif strategy == "cross_validation":
            if self.cross_validation_config is None:
                raise ValueError(
                    "`cross_validation_config` must be provided for 'cross_validation' strategy."
                )
        return self

    @model_validator(mode="after")
    def enforce_naming_convention(self) -> "ModelTrainerConfig":
        """
        Ensure the configuration dictionary key matches the internal model type.

        Raises:
            ValueError: If a dictionary key (e.g., 'LGBM') does not match
                the 'type' field in its parameters.
        """
        for key, params in self.params.items():
            if key != params.type:
                raise ValueError(
                    f"Configuration Error: The Config Key '{key}' does not match "
                    f"the internal Model Type '{params.type}'.\n"
                    f"Current policy requires them to match."
                )
        return self


class ModelRegistryConfig(BaseModel):
    """Configuration for the model registration and versioning stage."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether to register the trained model in the MLflow Model Registry.",
    )

    target_alias: str = Field(
        default="staging",
        description="The alias tag to apply after registration (e.g. 'staging', 'candidate').",
    )


class PredictionConfig(BaseModel):
    """Configuration for the prediction/inference service."""

    model_config = ConfigDict(extra="forbid")
    default_model_name: str
    model_filename: str
    artifacts: ArtifactsConfig
    mlflow: MlflowConfig


# =============================================================================
# 4. THE ROOT SCHEMA
# =============================================================================
class ConfigSchema(BaseModel):
    """The root configuration schema for the entire ML pipeline."""

    model_config = ConfigDict(extra="forbid")

    globals: GlobalsConfig
    data_ingestion: DataIngestionConfig
    data_preparation: DataPreparationConfig
    model_trainer: ModelTrainerConfig
    model_registry: ModelRegistryConfig
    prediction: PredictionConfig
