"""Data loading and preprocessing utilities for coffee productivity analysis."""

import errno
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

data_dtype = np.float32
neg_inf = -data_dtype(np.inf)
pos_inf = data_dtype(np.inf)
accumulator_dtype = np.float64

AnyShapeDataDtypeArray = TypeVar("AnyShapeDataDtypeArray", bound=np.ndarray[tuple[int, ...], np.dtype[data_dtype]])


class RequiredColumn(StrEnum):
    """Required Columns in the coffe productivity CSV data."""

    CUPS = "cups"
    PRODUCTIVITY = "productivity"


class CSVReadError(RuntimeError):
    """Exception for data input failure."""

    def __init__(self, filepath: Path) -> None:
        super().__init__(f"Failed to read CSV file: {filepath}")


class MissingColumnsError(ValueError):
    """Exception for missing required columns in data."""

    def __init__(self, missing_columns: set[RequiredColumn]) -> None:
        missing_columns_str = {col.value for col in missing_columns}
        super().__init__(f"Missing required columns: {missing_columns_str}")


class ColumnTypeError(ValueError):
    """Exception for invalid column type."""

    def __init__(self, col: RequiredColumn, dtype: np.dtype) -> None:
        super().__init__(f"Column {col.value} must contain numeric values, but found dtype {dtype}")


def validate(data: pd.DataFrame) -> None:
    """Validate `data` content by checking column presence and types.

    Parameters
    ----------
    data : pd.DataFrame
        Panda Dataframe to validate.

    Raises
    ------
    MissingColumnsError
        If a required column is missing from the DataFrame.
    ColumnTypeError
        If a required column has an invalid type. Required columns are expected to have a numerical type.
    """
    # Check for missing columns
    missing_columns = {col for col in RequiredColumn if col.value not in data.columns}
    if missing_columns:
        raise MissingColumnsError(missing_columns)
    # Check for invalid column type.
    for col in RequiredColumn:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ColumnTypeError(col, data.dtypes[col])


def curate(data: pd.DataFrame) -> pd.DataFrame:
    """Curate `data` by removing rows with `NaN` values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame content to curate.

    Returns
    -------
    pd.DataFrame
        The curated DataFrame, possibly with removed rows.
    """
    # Remove any rows with NaN values
    initial_size = len(data)
    curated_data = data.dropna(subset=list(RequiredColumn)).reset_index(drop=True)
    if len(data) < initial_size:
        logger = logging.getLogger(__name__)
        logger.warning("Dropped %s rows due to NaN values", initial_size - len(data))
    return curated_data


def load_csv(filepath: Path) -> pd.DataFrame:
    r"""Load coffee productivity data from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing the data.
        Expected columns: 'cups' (int) and 'productivity' (float).

    Returns
    -------
    pd.DataFrame
        DataFrame with validated columns 'cups' and 'productivity'.

    Raises
    ------
    CSVReadError
        If the CSV reading fails
    ColumnTypeError
        If required columns contain invalid data.
    FileNotFoundError
        If the specified data file does not exist.
    MissingColumnsError
        If required columns are missing.

    Examples
    --------
    >>> prod_data = load_csv(Path("coffee_productivity.csv"))  # doctest: +SKIP
    >>> print(prod_data.head())  # doctest: +SKIP
       cups  productivity
    0     1           2.1
    """
    if not filepath.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        raise CSVReadError(filepath) from e

    validate(data)
    return curate(data)


def extract_arrays(
    data: pd.DataFrame,
) -> tuple[np.ndarray[tuple[int], np.dtype[data_dtype]], np.ndarray[tuple[int], np.dtype[data_dtype]]]:
    """Extract cups and productivity as numpy arrays from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'cups' and 'productivity' columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (cups, productivity) as float arrays.

    Examples
    --------
    >>> data = pd.DataFrame({"cups": [1, 3, 5], "productivity": [0.3, 1.5, 0.8]})
    >>> cups, productivity = extract_arrays(data)
    >>> print(cups.shape, productivity.shape)
    (3,) (3,)
    """
    cups = data[RequiredColumn.CUPS].to_numpy(dtype=data_dtype)
    productivity = data[RequiredColumn.PRODUCTIVITY].to_numpy(dtype=data_dtype)
    return cups, productivity
