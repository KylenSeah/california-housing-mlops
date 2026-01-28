"""
Data schema definitions for the California Housing MLOps pipeline.

This module leverages Pandera to enforce strict data contracts, ensuring
that all incoming data adheres to specified types, ranges, and categorical
constraints before entering the training or inference pipelines.
"""

import pandera.pandas as pa
from pandera.typing import Series

# =============================================================================
# HOUSING SCHEMA DEFINITION
# =============================================================================


class HousingSchema(pa.DataFrameModel):
    """
    Data validation contract for the California Housing dataset.

    This class defines the required columns, data types, and physical
    constraints for the raw input data. It is used as a 'Quality Gate'
    to ensure data integrity before training.
    """

    longitude: Series[float] = pa.Field(
        ge=-180, le=180, nullable=True, description="Longitude of the house location."
    )
    latitude: Series[float] = pa.Field(
        ge=-90, le=90, nullable=True, description="Latitude of the house location."
    )
    housing_median_age: Series[float] = pa.Field(
        gt=0, nullable=True, description="Median age of houses in the block."
    )
    total_rooms: Series[float] = pa.Field(
        gt=0, nullable=True, description="Total number of rooms in the block."
    )
    total_bedrooms: Series[float] = pa.Field(
        gt=0, nullable=True, description="Total number of bedrooms in the block."
    )
    population: Series[float] = pa.Field(
        gt=0, nullable=True, description="Total population in the block."
    )
    households: Series[float] = pa.Field(
        gt=0, nullable=True, description="Total number of households in the block."
    )
    median_income: Series[float] = pa.Field(
        gt=0, nullable=True, description="Median income for households in the block."
    )
    median_house_value: Series[float] = pa.Field(
        gt=0,
        nullable=True,
        description="Median house value for households in the block.",
    )
    ocean_proximity: Series[str] = pa.Field(
        isin=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
        nullable=True,
        description="Proximity to the ocean.",
    )

    class Config:  # type: ignore
        """
        Validation engine settings.

        Strictness is enabled to prevent unexpected columns from polluting
        the training set, while coercion allows for automatic type casting.
        """

        strict = True
        ordered = False
        coerce = True
