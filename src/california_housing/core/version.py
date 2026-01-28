"""
Version metadata for the California Housing MLOps package.

This module provides programmatic access to the version string defined in the
project's pyproject.toml. It leverages importlib.metadata for high-reliability
version retrieval in installed environments.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("california-housing-mlops")
except PackageNotFoundError:
    __version__ = "0.0.0"
