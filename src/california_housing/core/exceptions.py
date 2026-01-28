"""
Domain-specific exceptions for the California Housing MLOps pipeline.

This module defines a hierarchy of custom exceptions to enable granular
error handling and clear telemetry tagging across the system.
"""

# =============================================================================
# INFRASTRUCTURE & IO ERRORS
# =============================================================================


class IngestionError(Exception):
    """Raised when the data source cannot be accessed or read (Infrastructure/IO)."""

    pass


class TelemetryError(Exception):
    """
    Raised when failing to log metrics, artifacts, or traces.
    Usually treated as non-blocking/best-effort in inference.
    """

    pass


# =============================================================================
# DATA QUALITY & VALIDATION ERRORS
# =============================================================================


class DataValidationError(Exception):
    """Raised when the data itself fails quality checks or statistical requirements."""

    pass


class ConfigurationError(Exception):
    """Raised when the application contract (config.yaml) is violated."""

    pass


# =============================================================================
# MODEL LIFECYCLE ERRORS
# =============================================================================


class ModelRegistryError(Exception):
    """Raised when MLflow model registry operations fail."""

    pass


class ModelTrainingError(Exception):
    """Raised when model training or internal validation fails."""

    pass
