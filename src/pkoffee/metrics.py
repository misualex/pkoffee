"""Model evaluation metrics for assessing fit quality."""

import numpy as np

from pkoffee.data import accumulator_dtype, data_dtype


class SizeMismatchError(ValueError):
    """Exception for data input failure."""

    def __init__(self, size_a: int, size_b: int) -> None:
        super().__init__(f"Arrays must have same length, got {size_a} and {size_b}")


def check_size_match(array_a: np.ndarray, array_b: np.ndarray) -> None:
    """Check that 2 array have the same size, throw SizeMismatchError if not.

    Parameters
    ----------
    array_a : np.ndarray
        First array
    array_b : np.ndarray
        Second array

    Raises
    ------
    SizeMismatchError
        If the two arrays sizes aren't equal.
    """
    if len(array_a) != len(array_b):
        raise SizeMismatchError(len(array_a), len(array_b))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> data_dtype:
    """Calculate the coefficient of determination (R² score).

    R² indicates the proportion of variance in the dependent variable
    that is predictable from the independent variable(s).

    Parameters
    ----------
    y_true : np.ndarray
        True observed values.
    y_pred : np.ndarray
        Predicted values from the model.

    Returns
    -------
    float
        R² score. Values closer to 1.0 indicate better fit.
        Can be negative for very poor fits.

    Notes
    -----
    R² = 1 - (SS_res / SS_tot)
    where:
        SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
        SS_tot = Σ(y_true - ȳ)²       (total sum of squares)

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    >>> r2 = compute_r2(y_true, y_pred)
    >>> print(f"R² = {r2:.4f}")
    R² = 0.9920
    """
    y_true = np.asarray(y_true, dtype=data_dtype)
    y_pred = np.asarray(y_pred, dtype=data_dtype)

    check_size_match(y_true, y_pred)

    residual_sum_of_squares = data_dtype(np.sum((y_true - y_pred) ** 2, dtype=accumulator_dtype))
    total_sum_of_squares = data_dtype(np.sum((y_true - np.mean(y_true)) ** 2, dtype=accumulator_dtype))

    if total_sum_of_squares == 0:
        return data_dtype(np.nan)

    return data_dtype(1.0 - (residual_sum_of_squares / total_sum_of_squares))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> data_dtype:
    """
    Calculate Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        True observed values.
    y_pred : np.ndarray
        Predicted values from the model.

    Returns
    -------
    float
        RMSE value. Lower is better, 0 is perfect fit.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    >>> rmse = compute_rmse(y_true, y_pred)
    >>> print(f"RMSE = {rmse:.4f}")
    RMSE = 0.1000
    """
    y_true = np.asarray(y_true, dtype=data_dtype)
    y_pred = np.asarray(y_pred, dtype=data_dtype)

    check_size_match(y_true, y_pred)

    mse = np.mean((y_true - y_pred) ** 2, dtype=accumulator_dtype)
    return data_dtype(np.sqrt(mse))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> data_dtype:
    """Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        True observed values.
    y_pred : np.ndarray
        Predicted values from the model.

    Returns
    -------
    float
        MAE value. Lower is better, 0 is perfect fit.

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    >>> mae = compute_mae(y_true, y_pred)
    >>> print(f"MAE = {mae:.4f}")
    MAE = 0.1000
    """
    y_true = np.asarray(y_true, dtype=data_dtype)
    y_pred = np.asarray(y_pred, dtype=data_dtype)

    check_size_match(y_true, y_pred)

    return data_dtype(np.mean(np.abs(y_true - y_pred), dtype=accumulator_dtype))
