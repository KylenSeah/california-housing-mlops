"""
Modular preprocessing and feature engineering engine.

This module provides custom Scikit-Learn transformers and a centralized factory
function to build the complete data transformation pipeline. It handles
imputation, scaling, encoding, and advanced geospatial/ratio-based engineering.
"""

import logging
import re
from typing import Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore

from california_housing.core.config_definitions import (
    CategoryMergingConfig,
    PipelinePreprocessingConfig,
)

# Type alias for inputs that Scikit-Learn transformers accept
ArrayLike: TypeAlias = np.ndarray | pd.DataFrame

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM TRANSFORMERS (Engineering Primitives)
# =============================================================================


class FeatureNameSanitizer(BaseEstimator, TransformerMixin):
    """
    Sanitize column names to ensure compatibility with downstream models.

    LightGBM and XGBoost often crash or produce warnings if feature names
    contain special characters (like '[' or '<') or spaces. This transformer
    ensures all feature names follow the 'snake_case' pattern.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: ArrayLike, y: pd.Series | None = None) -> "FeatureNameSanitizer":
        """Stateless transformer; returns self."""
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Apply sanitized feature names to the input matrix, with fallback for RangeIndex columns."""
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            X_transformed = pd.DataFrame(X)

        if isinstance(X_transformed.columns, pd.RangeIndex):
            logger.warning("No input columns provided; using default indexing.")
            input_cols = [f"feature_{i}" for i in X_transformed.columns]
        else:
            input_cols = X_transformed.columns.tolist()

        X_transformed.columns = self.get_feature_names_out(input_cols)
        return X_transformed

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Apply regex sanitization to replace non-alphanumeric characters with underscores."""
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out")

        sanitized_feature = [
            re.sub(r"[^a-zA-Z0-9_]", "_", str(col)) for col in input_features
        ]
        return sanitized_feature

    def __sklearn_is_fitted__(self) -> bool:
        """Inform Scikit-Learn that this transformer is always ready."""
        return True


class CategoryMerger(BaseEstimator, TransformerMixin):
    """
    Merges specific categorical values into a single category.

    Useful for handling rare categories (e.g., merging 'ISLAND' into 'NEAR OCEAN')
    to prevent model instability or creation of sparse columns during One-Hot Encoding.
    It prioritizes explicit column name lookup to ensure safety.
    """

    def __init__(self, col: str, mapping: dict[str, str]) -> None:
        """
        Initialize the merger.

        Args:
            col (str): The name of the categorical column to modify.
            mapping (dict[str, str]): A dictionary where keys are the old values
                                      and values are the new replacement values.
        """
        super().__init__()
        self.col = col
        self.mapping = mapping

        self.col_idx_: int | None = None
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoryMerger":
        """
        Validate input and mark as fitted.

        This is a stateless transformer (it does not learn from data), but
        it validates that the input has the required structure.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series | None): Ignored. Exists for compatibility.

        Returns:
            self: The fitted transformer.
        """
        _ = validate_data(self, X, dtype=None, ensure_min_features=1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the value mapping to the specified column.

        Args:
            X (pd.DataFrame): Input data. Must be a DataFrame to allow name-based lookup.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If the target column is missing from X.

        Returns:
            pd.DataFrame: The transformed DataFrame with merged categories.
        """
        # 1. Enforce the Contract (Fail Fast)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"CategoryMerger received {type(X)} but expects pd.DataFrame. "
                "Ensure this transformer is placed before steps that output NumPy arrays "
                "(e.g., before SimpleImputer)."
            )

        # 2. Check Existence
        if self.col not in X.columns:
            raise ValueError(f"Column '{self.col}' not found in input DataFrame.")

        X_out = X.copy()

        # 3. Apply Mapping
        X_out[self.col] = X_out[self.col].replace(self.mapping)

        return X_out

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """
        Return the feature names after transformation (unchanged).

        Args:
            input_features (list[str] | None): Input feature names.

        Returns:
            list[str]: The list of output feature names.
        """
        if input_features is not None:
            return input_features

        if self.feature_names_in_ is not None:
            return list(self.feature_names_in_)
        raise ValueError(
            f"{self.__class__.__name__} cannot determine output feature names. "
            "Ensure the transformer was fitted on a DataFrame."
        )

    def __sklearn_is_fitted__(self) -> bool:
        """Return True since this transformer requires no learning."""
        return True


class RatioFeature(BaseEstimator, TransformerMixin):
    """
    Compute ratios between two existing features to capture density signals.

    For example, 'bedrooms per room' often provides a stronger economic
    signal than raw counts of bedrooms or rooms.
    """

    def __init__(self, col_a: str, col_b: str) -> None:
        """
        Args:
            col_a (str): The numerator column name.
            col_b (str): The denominator column name.
        """
        super().__init__()
        self.col_a = col_a
        self.col_b = col_b

        self.n_features_in_: int | None = None
        self.col_a_idx_: int | None = None
        self.col_b_idx_: int | None = None

    def fit(self, X: ArrayLike, y: pd.Series | None = None) -> "RatioFeature":
        """
        Verify that exactly two features are provided for the ratio calculation.

        Args:
            X (ArrayLike): The input data containing exactly two columns.

        Raises:
            ValueError: If the feature count is not exactly 2.
        """
        X = validate_data(self, X, ensure_min_features=2)

        if self.n_features_in_ != 2:
            raise ValueError(
                f"Expected exactly 2 features, but got {self.n_features_in_}"
            )

        self.col_a_idx_ = 0
        self.col_b_idx_ = 1
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Execute ratio calculation with vectorized safety checks.

        Suppresses numpy division warnings and converts non-finite results
        (resulting from division by zero or NaN denominators) to NaNs for
        downstream imputation.

        Returns:
            np.ndarray: A single-column matrix of calculated ratios.
        """
        check_is_fitted(self)

        if self.col_a_idx_ is None or self.col_b_idx_ is None:
            raise RuntimeError("Transformer not fitted correctly")

        X = cast(np.ndarray, validate_data(self, X, reset=False))

        num = X[:, self.col_a_idx_]
        denom = X[:, self.col_b_idx_]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_col = np.true_divide(num, denom)

            mask_invalid = ~np.isfinite(ratio_col)

            if np.any(mask_invalid):
                logger.debug(
                    "Computed ratio %s/%s produced %s non-finite values",
                    self.col_a,
                    self.col_b,
                    np.sum(mask_invalid),
                )
                ratio_col[mask_invalid] = np.nan
        return cast(np.ndarray, ratio_col.reshape(-1, 1))

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Define the name of the engineered ratio feature."""
        return [f"{self.col_a}_per_{self.col_b}"]


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Measure similarity to geographical clusters using RBF kernels.

    This transformer turns raw coordinates (Lat/Lon) into 'distance-based'
    features. By using value-weighted K-Means during fitting, it identifies
    proximity to major economic hotspots (wealth hubs) rather than
    mere population density centers.
    """

    def __init__(
        self,
        n_clusters: int,
        gamma: float,
        random_state: int | None = None,
    ) -> None:
        """Initialize with KMeans cluster count and RBF kernel width."""
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        super().__init__()

        self.kmeans_: KMeans | None = None

    def fit(self, X: ArrayLike, y: pd.Series | None = None) -> "ClusterSimilarity":
        """
        Fit the KMeans model on geographical coordinates.

        Args:
            X: Geographical coordinates (Longitude, Latitude).
            y: Target values (House Prices) used to weight cluster centers
               toward high-value economic hotspots.
        """
        X = validate_data(self, X)

        if y is None:
            logger.warning(
                "ClusterSimilarity.fit received y=None. Falling back to unweighted KMeans."
            )
            sample_weight = None
        else:
            sample_weight = np.array(y).flatten()

            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight (y) contains NaNs or infinite values.")

            if np.any(sample_weight < 0):
                raise ValueError("sample_weight (y) cannot contain negative values.")

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=y)
        logger.info(
            "Fitted KMeans with %s clusters (random state=%s, Weighted=%s)",
            self.n_clusters,
            self.random_state,
            "Yes" if sample_weight is not None else "No",
        )
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform coordinates into cluster similarity scores."""
        check_is_fitted(self)

        if self.kmeans_ is None:
            raise RuntimeError("The KMeans estimator has not been fitted")

        X = validate_data(self, X, reset=False)

        return cast(
            np.ndarray, rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
        )

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Generate names for each cluster similarity feature."""
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]


# =============================================================================
# PIPELINE FACTORY (The Blueprint)
# =============================================================================


def create_preprocessing_pipeline(
    config: PipelinePreprocessingConfig,
    random_state: int,
) -> Pipeline:
    """
    Construct the full Scikit-Learn preprocessing pipeline.

    This factory builds a ColumnTransformer that orchestrates multiple
    sub-pipelines (Log, Geo, Categorical, Ratios) based on the provided
    configuration. Fitting is strictly isolated to the training set.

    Args:
        config (PipelinePreprocessingConfig): Blueprint for the transformations.
        random_state (int): Global seed for stochastic operations (KMeans).

    Returns:
        Pipeline: A fully assembled, Pandas-out compatible Pipeline.
    """
    logger.info("Building preprocessing pipeline from configuration.")

    # 1. Component Construction
    ratio_cols = config.ratio_cols
    log_cols = config.log_cols
    geo_cols = config.geo_cols
    n_clusters = config.n_clusters
    gamma = config.gamma
    imputer_strategy_num = config.imputer_strategy_num
    imputer_strategy_cat = config.imputer_strategy_cat
    onehot_handle_unknown = config.onehot_handle_unknown
    log_pipeline = _build_log_pipeline(imputer_strategy=imputer_strategy_num)
    geo_pipeline = _build_geo_pipeline(
        n_clusters=n_clusters,
        gamma=gamma,
        random_state=random_state,  # Use the globally passed random_state
        imputer_strategy=imputer_strategy_num,
    )
    cat_pipeline = _build_cat_pipeline(
        imputer_strategy=imputer_strategy_cat,
        handle_unknown=onehot_handle_unknown,
        merge_config=config.category_merging,
    )
    default_num_pipeline = _build_default_num_pipeline(
        imputer_strategy=imputer_strategy_num
    )
    ratio_transformers = _build_ratio_transformers(
        ratio_cols=ratio_cols, imputer_strategy=imputer_strategy_num
    )

    # 2. Composition (Column-level)
    column_transformer = ColumnTransformer(
        transformers=ratio_transformers
        + [
            ("log", log_pipeline, log_cols),
            ("geo", geo_pipeline, geo_cols),
            (
                "cat",
                cat_pipeline,
                make_column_selector(dtype_include=np.dtype("object")),
            ),
        ],
        remainder=default_num_pipeline,
    )

    # 3. Final Orchestration (Pipeline-level)
    steps: list[tuple[str, BaseEstimator]] = [
        ("column_transformer", column_transformer),
        ("sanitizer", FeatureNameSanitizer()),
    ]

    # Optional Cleanups
    cleanup = config.final_cleanup
    if cleanup.handle_inf:
        steps.append(
            (
                "inf_to_nan",
                FunctionTransformer(
                    lambda X: np.where(np.isinf(X), np.nan, X),
                    feature_names_out="one-to-one",
                ),
            )
        )

    if cleanup.perform_imputation:
        steps.append(
            ("final_imputer", SimpleImputer(strategy=cleanup.imputer_strategy))
        )

    preprocessor = Pipeline(steps)
    preprocessor.set_output(transform="pandas")
    logger.info(
        "Preprocessing pipeline built. Final cleanup: Inf=%s, Impute=%s (strategy=%s)",
        cleanup.handle_inf,
        cleanup.perform_imputation,
        cleanup.imputer_strategy,
    )
    return preprocessor


def _build_log_pipeline(imputer_strategy: str) -> Pipeline:
    """Helper to build log transformation pipeline with configurable imputer."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_geo_pipeline(
    n_clusters: int, gamma: float, random_state: int, imputer_strategy: str
) -> Pipeline:
    """Helper to build geographic clustering pipeline with configurable params."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler_input", StandardScaler()),
            (
                "cluster_simil",
                ClusterSimilarity(
                    n_clusters=n_clusters, gamma=gamma, random_state=random_state
                ),
            ),
            ("scaler_output", StandardScaler()),
        ]
    )


def _build_cat_pipeline(
    imputer_strategy: str,
    handle_unknown: Literal["error", "ignore", "infrequent_if_exist"],
    merge_config: CategoryMergingConfig | None,
) -> Pipeline:
    """
    Helper to build categorical pipeline with optional category merging.

    If merge_config is provided, the CategoryMerger is inserted as the FIRST step.
    This ensures that rare categories are mapped (e.g., ISLAND -> NEAR OCEAN)
    BEFORE the OneHotEncoder sees them.
    """

    steps = []

    # 1. Optional: Merge Categories (MUST happen before imputation/encoding)
    if merge_config is not None:
        steps.append(
            (
                "category_merger",
                CategoryMerger(col=merge_config.col, mapping=merge_config.mapping),
            )
        )

    # 2. Imputation & Encoding
    steps.extend(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            (
                "one_hot",
                OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False),
            ),
        ]
    )
    return Pipeline(steps=steps)


def _build_default_num_pipeline(imputer_strategy: str) -> Pipeline:
    """Helper to build default numerical pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", StandardScaler()),
        ]
    )


def _build_ratio_transformers(
    ratio_cols: dict[str, tuple[str, str]],
    imputer_strategy: str,
) -> list[tuple[str, Pipeline, list[str]]]:
    """Generate a list of Scikit-Learn sub-pipelines for engineering ratio-based features."""
    return [
        (
            name,
            Pipeline(
                [
                    ("imputer_input", SimpleImputer(strategy=imputer_strategy)),
                    ("ratio", RatioFeature(col_a=cols[0], col_b=cols[1])),
                    ("imputer_ratio", SimpleImputer(strategy=imputer_strategy)),
                    ("scaler", StandardScaler()),
                ]
            ),
            list(cols),
        )
        for name, cols in ratio_cols.items()
    ]
