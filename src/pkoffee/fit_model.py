"""Mathematical models for coffee productivity relationships.

This module provides various parametric models that can be fitted to
coffee consumption vs productivity data.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.optimize import curve_fit

from pkoffee.data import AnyShapeDataDtypeArray, data_dtype
from pkoffee.metrics import compute_r2
from pkoffee.parametric_function import (
    ParametersBounds,
    ParametricFunction,
)


class FunctionNotFoundInMappingError(KeyError):
    """Exception when a function is not found in the function to str mapping."""

    def __init__(self, function: type[ParametricFunction], mapping: Mapping) -> None:
        super().__init__(f"Function {function} not found in function to str mapping {mapping}")


class FunctionIdNotFoundInMappingError(KeyError):
    """Exception when a function Identifier is not found in the function Id to function mapping."""

    def __init__(self, function_id: str, mapping: Mapping) -> None:
        super().__init__(f"Function Identifier {function_id} not found in mapping to function {mapping}")


class ModelParsingError(ValueError):
    """Exception when a model dictionary representation can not be parsed into a model."""

    def __init__(self, model_dict: Mapping) -> None:
        super().__init__(f"Could not parse model dictionary {model_dict}, missing fields or bad types?")


@dataclass
class Model:
    """Model defined by a prediction function, parameters and parameter's bounds.

    Attributes
    ----------
    name : str
        Name of the model
    function : ParametricFunction
        The model prediction function
    params : tuple[data_types]
        Model parameters passed to the predict function
    bounds : ParametersBounds
        Boundary values for the model parameters.
    """

    name: str
    function: ParametricFunction
    params: dict[str, data_dtype]
    bounds: ParametersBounds
    r_squared: data_dtype = -data_dtype(np.inf)

    def predict(self, x: AnyShapeDataDtypeArray) -> AnyShapeDataDtypeArray:
        """Evaluate the model on input `x`.

        Parameters
        ----------
        x : np.ndarray
            The model input as 1D array.

        Returns
        -------
        np.ndarray
            Prediction of the model, same shape as `x`.
        """
        return self.function(x, **self.params)

    def __repr__(self) -> str:
        """Return a formatted string representation of the model result."""
        return f"ModelFit(name='{self.name}', R²={self.r_squared:.3f})"

    @classmethod
    def sort(cls, models: list[Self]) -> None:
        """Sort `models` by R² (descending), in-place.

        Parameters
        ----------
        models : list[Self]
            List of models to sort by R²
        """
        models.sort(
            key=lambda r: r.r_squared if np.isfinite(r.r_squared) else -np.inf,
            reverse=True,
        )

    def to_dict(self, function_to_str: Mapping) -> dict:
        """Convert model to pure python dictionary representation of the model.

        Numbers are converted to python's floats, and the function is encoded as a string according to
        `function_to_str`.

        Parameters
        ----------
        function_to_str : Mapping
            Dict mapping function classes to a string identifier. Ex: {pkoffee.fit_model.Quadratic: "quadratic"}

        Returns
        -------
        dict
            Dictionary representation of a model.

        Examples
        --------
        >>> from pkoffee.data import data_dtype
        >>> from pkoffee.fit_model_io import pkoffee_function_id_mapping, Quadratic
        >>> def_quad = Model(
        ...     name="DefaultQuadratic",
        ...     function=Quadratic(),
        ...     params=Quadratic.param_guess(y_min=data_dtype(0.5)),
        ...     bounds=Quadratic.param_bounds(),
        ... )
        >>> def_quad.to_dict(pkoffee_function_id_mapping().inv)
        {'name': 'DefaultQuadratic', 'function': 'Quadratic', 'params': {'a0': 0.5, 'a1': 0.0, 'a2': 0.009999999776482582}, 'bounds': {'min': {'a0': -inf, 'a1': -inf, 'a2': -inf}, 'max': {'a0': inf, 'a1': inf, 'a2': inf}}, 'r_squared': -inf}
        """  # noqa: E501 doctest string is too large
        try:
            return {
                "name": self.name,
                "function": function_to_str[type(self.function)],
                "params": {p: float(v) for p, v in self.params.items()},
                "bounds": {
                    "min": {p: float(v) for p, v in self.bounds.min.items()},
                    "max": {p: float(v) for p, v in self.bounds.max.items()},
                },
                "r_squared": float(self.r_squared),
            }
        except KeyError as e:
            raise FunctionNotFoundInMappingError(type(self.function), function_to_str) from e

    @classmethod
    def from_dict(cls, d: Mapping, str_to_function: Mapping) -> Self:
        """Create a model from a dictionary representation.

        Parameters
        ----------
        d : Mapping
            Mapping representation of a Model as return by `Model.to_dict`
        str_to_function : Mapping
            Mapping function identifiers to actual function classes

        Returns
        -------
        Self
            Model instance

        Examples
        --------
        >>> from pkoffee.fit_model_io import pkoffee_function_id_mapping
        >>> Model.from_dict(
        ...     {
        ...         "name": "TestQuadratic",
        ...         "function": "Quadratic",
        ...         "params": {"a": 1.0, "b": 0.0, "c": 0.5},
        ...         "bounds": {"min": {"a": -5.0, "b": -2.0, "c": -1.0}, "max": {"a": 5.0, "b": 2.0, "c": 1.0}},
        ...         "r_squared": 0.22,
        ...     },
        ...     pkoffee_function_id_mapping(),
        ... )
        ModelFit(name='TestQuadratic', R²=0.220)
        """
        try:
            function_id = d["function"]
            try:
                param_function = str_to_function[function_id]()
            except KeyError as e:
                raise FunctionIdNotFoundInMappingError(d["function"], str_to_function) from e
            return cls(
                name=d["name"],
                function=param_function,
                params={p: data_dtype(v) for p, v in d["params"].items()},
                bounds=ParametersBounds(
                    min={p: data_dtype(v) for p, v in d["bounds"]["min"].items()},
                    max={p: data_dtype(v) for p, v in d["bounds"]["max"].items()},
                ),
                r_squared=data_dtype(d["r_squared"]),
            )
        except FunctionIdNotFoundInMappingError:
            raise
        except (KeyError, ValueError) as e:
            raise ModelParsingError(d) from e


def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    model: Model,
    max_iterations: int = 20000,
) -> tuple[Model, np.ndarray]:
    """Fit a single model to the data.

    Parameters
    ----------
    x : np.ndarray
        Input data (independent variable)
    y : np.ndarray
        Output data (dependent variable)
    model : Model
        Model including function and parameters
    max_iterations : int, optional
        Maximum number of optimization iterations, by default 20000

    Returns
    -------
    tuple[FittedModel, np.ndarray]
        tuple with Fitted model and predictions on training data

    Raises
    ------
    ValueError
        If either x or y contain NaNs.
    RuntimeError
        If the least-squares minimization fails.
    """
    params = model.params.keys()
    optimal_params, _ = curve_fit(
        model.function,
        x,
        y,
        p0=list(model.params.values()),  # same order as params
        bounds=(
            tuple(model.bounds.min[p] for p in params),
            tuple(model.bounds.max[p] for p in params),
        ),  # converting to tuples while respecting params order
        maxfev=max_iterations,
    )
    predictions = model.function(x, *optimal_params)
    r_squared = compute_r2(y, predictions)
    return (
        Model(
            model.name,
            function=model.function,
            params=dict(
                zip(model.params.keys(), optimal_params, strict=True)
            ),  # build back dictionary of parameters from values, the order is the same as the params of model
            bounds=model.bounds,
            r_squared=r_squared,
        ),
        predictions,
    )
