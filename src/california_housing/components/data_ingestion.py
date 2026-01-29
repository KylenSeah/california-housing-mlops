"""
Data ingestion component for the California Housing pipeline.

This module provides the DataIngestion class, which is responsible for
extracting raw data from the configured SQLite source and enforcing the
initial data contract via Pandera schemas.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from california_housing.core.config_definitions import DataIngestionConfig
from california_housing.core.data_definitions import HousingSchema
from california_housing.core.exceptions import DataValidationError, IngestionError

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handle the extraction and initial validation of raw housing data.

    This component acts as the physical 'Inlet' of the pipeline. It establishes
    connections to the data source, performs the extraction, and ensures the
    data is not empty and matches the expected schema.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize the DataIngestion component.

        Args:
            config (DataIngestionConfig): Validated configuration block
                containing 'db_path' and 'table_name'.
        """
        self.config = config

    def get_data(self) -> pd.DataFrame:
        """
        Orchestrate the loading and validation of housing data.

        This method acts as the high-level entry point for data acquisition,
        mapping configuration types to specific loading mechanisms.

        Raises:
            IngestionError: If the underlying infrastructure (file/DB) fails
                or if the source type is unsupported.
            DataValidationError: If the data is empty or violates the
                Pandera schema (Data Contract).

        Returns:
            pd.DataFrame: The schema-validated housing dataset.
        """
        # ---------------------------------------------------------
        # 1. MECHANISM: Try to load the raw bytes/rows
        # ---------------------------------------------------------
        try:
            if self.config.type == "sqlite":
                df = self._load_from_sqlite(
                    db_path=self.config.db_path,
                    table_name=self.config.table_name,
                )
            else:
                raise ValueError(f"Unsupported data source type: {self.config.type}")
        except (FileNotFoundError, sqlite3.Error, ValueError) as e:
            raise IngestionError(
                f"Failed to load data from source '{self.config.type}': {e}"
            ) from e

        # ---------------------------------------------------------
        # 2. POLICY: Check if the data is usable for OUR purpose
        # ---------------------------------------------------------
        if df.empty:
            msg = (
                f"Data Quality Violation: Table '{self.config.table_name}' is empty."
                "Training pipeline requires at least 1 row to proceed."
            )
            raise DataValidationError(msg)

        # ---------------------------------------------------------
        # 3. VALIDATION: Check Schema
        # ---------------------------------------------------------
        logger.info("Validating schema for %s rows...", len(df))
        df = HousingSchema.validate(df, lazy=True)
        logger.info("Schema validation passed.")
        return df

    def _load_from_sqlite(self, db_path: Path, table_name: str) -> pd.DataFrame:
        """
        Execute low-level SQLite extraction logic.

        Connects to the database file, verifies table existence via the
        master schema, and performs a full table read into memory.

        Args:
            db_path (Path): The filesystem path to the .db file.
            table_name (str): The specific SQL table to query.

        Raises:
            FileNotFoundError: If the database file does not exist at the provided path.
            ValueError: If the requested table name is missing from the database.

        Returns:
            pd.DataFrame: The raw DataFrame containing all rows from the table.
        """
        if not db_path.is_file():
            raise FileNotFoundError(f"Database file not found at path: {db_path}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )

            if cursor.fetchone() is None:
                raise ValueError(
                    f"Table '{table_name}' not found in database: {db_path}"
                )

            query = f"SELECT * FROM [{table_name}]"

            df = pd.read_sql_query(query, conn)

            return df
