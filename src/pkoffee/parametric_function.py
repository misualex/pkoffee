"""Parametric functions.

This module provides functions with signature f(x, *args, **kwargs), where `x` is the function's input and the other
arguments are the function parameters. Functions also provide guesses and boundaries for parameter values.
"""

from typing import NamedTuple, ParamSpec, Protocol, runtime_checkable

import numpy as np

from pkoffee.data import AnyShapeDataDtypeArray, data_dtype, neg_inf, pos_inf


class ParametersBounds(NamedTuple):
    """Store the minimum and maximum bounds.

    Attributes
    ----------
    min : dict[str, data_dtype]
        Minimum bounds
    max : dict[str, data_dtype]
        Maximum bounds
    """

    min: dict[str, data_dtype]
    max: dict[str, data_dtype]


AnyDataDtypeArgsKwargs = ParamSpec("AnyDataDtypeArgsKwargs", bound=data_dtype)  # pyright: ignore[reportGeneralTypeIssues] bound parameter exists.
AnyArgsKwargs = ParamSpec("AnyArgsKwargs")


@runtime_checkable
class ParametricFunction(Protocol[AnyShapeDataDtypeArray, AnyDataDtypeArgsKwargs, AnyArgsKwargs]):
    """Parametric function API."""

    def __call__(
        self, x: AnyShapeDataDtypeArray, *args: AnyDataDtypeArgsKwargs.args, **kwargs: AnyDataDtypeArgsKwargs.kwargs
    ) -> AnyShapeDataDtypeArray:
        """Evaluate the parametric function on input `x` with parameters values passed with `*args` and `**kwargs`.

        Parameters
        ----------
        x : AnyShapeDataDtypeArray
            The input to the function, an array of any shape, of dtype `data_dtype`.

        Returns
        -------
        AnyShapeDataDtypeArray
            Output of the function at `x`.
        """
        ...

    @classmethod
    def param_guess(cls, *args: AnyArgsKwargs.args, **kwargs: AnyArgsKwargs.kwargs) -> dict[str, data_dtype]:
        """Guess values of the `ParametricFunction` parameters.

        The guess values can typically be used as starting values for a fit of the parameters.

        The guesses may require some information about the data (eg. range, min/max values) therefore this method is
        allowed to take any input.

        Returns
        -------
        dict[str, data_dtype]
            Dictionary mapping parameter names to guessed values.
        """
        ...

    @classmethod
    def param_bounds(cls) -> ParametersBounds:
        """Min/max values of the `ParametricFunction` parameters.

        The `ParametersBound` dictionaries' keys are the parameters' names.

        Returns
        -------
        ParametersBounds
            min/max values of the parameters.
        """
        ...


class Quadratic:
    """Quadratic (polynomial) function: f(x) = a₀ + a₁x + a₂x².

    References
    ----------
    1. Wikipedia contributors. (2025, September 16). Quadratic function. In Wikipedia, The Free Encyclopedia.
    Retrieved 19:28, December 1, 2025,
    from https://en.wikipedia.org/w/index.php?title=Quadratic_function&oldid=1311755644
    """

    def __call__(
        self, x: AnyShapeDataDtypeArray, a0: data_dtype, a1: data_dtype, a2: data_dtype
    ) -> AnyShapeDataDtypeArray:
        """Evaluate the quadratic function at each point in `x`.

        Parameters
        ----------
        x : AnyShapeDataDtypeArray
            Input values
        a0 : data_dtype
            Constant term
        a1 : data_dtype
            Linear coefficient
        a2 : data_dtype
            Quadratic coefficient

        Returns
        -------
        AnyShapeDataDtypeArray
            QuadraticFunction value at each point in `x`.
        """
        return a0 + a1 * x + a2 * x**2  # type: ignore[reportReturnType] return type is data_dtype alright

    @classmethod
    def param_guess(cls, y_min: data_dtype) -> dict[str, data_dtype]:
        """Parameter guesses for a fit starting values.

        The linear coefficient guess is 0.0, and the quadratic coefficient 0.01. The constant term guess is the
        minimum value of the predictions in the data points: if modeling y = a₀ + a₁x + a₂x², then min(y).

        Parameters
        ----------
        y_min : data_dtype
            The minimal value of the predictions.

        Returns
        -------
        dict[str, data_dtype]
            Dictionary mapping parameter names to guesses.
        """
        return {"a0": y_min, "a1": data_dtype(0.0), "a2": data_dtype(0.01)}

    @classmethod
    def param_bounds(cls) -> ParametersBounds:
        """Boundary values for the `QuadraticFunction."""
        params = ["a0", "a1", "a2"]
        return ParametersBounds(min=dict.fromkeys(params, neg_inf), max=dict.fromkeys(params, pos_inf))


class MichaelisMentenSaturation:
    """Michaelis-Menten (saturating) model: f(x) = y₀ + Vₘₐₓ·x/(K + x).

    This model describes saturation behavior common in enzyme kinetics
    and can represent diminishing returns.

    References
    ----------
    1. Wikipedia contributors. (2025, December 1). Michaelis-Menten kinetics. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:32, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Michaelis%E2%80%93Menten_kinetics&oldid=1325118298
    """

    def __call__(
        self, x: AnyShapeDataDtypeArray, v_max: data_dtype, k: data_dtype, y0: data_dtype
    ) -> AnyShapeDataDtypeArray:
        """Evaluate the MichaelisMenten Function in each point in `x`.

        Parameters
        ----------
        x : AnyShapeDataDtypeArray
            Input values
        v_max : data_dtype
            Maximum rate/value
        k : data_dtype
            Half-saturation constant (Michaelis constant)
        y0 : data_dtype
            Baseline offset

        Returns
        -------
        AnyShapeDataDtypeArray
            MichaelisMenten function value at each point in `x`.
        """
        return y0 + v_max * (x / np.maximum(k + x, 1e-9))  # type: ignore[reportReturnType] return type is data_dtype alright

    @classmethod
    def param_guess(
        cls, x_min: data_dtype, x_max: data_dtype, y_min: data_dtype, y_max: data_dtype
    ) -> dict[str, data_dtype]:
        """Parameter guesses for a fit initial values.

        x are the function input values, y the predictions in the data points. `v_max` guess is the prediction range,
        `k` the input value at mid-growth is guessed as the input value at 20% of the input range, `y0`'s guess is the
        minimum input value.

        Parameters
        ----------
        x_min : data_dtype
            Maximum input value
        x_max : data_dtype
            Maximum input value
        y_min : data_dtype
            Minimum prediction value
        y_max : data_dtype
            Maximum prediction value

        Returns
        -------
        dict[str, data_dtype]
            Dictionary mapping parameter names to guesses.
        """
        return {
            "v_max": max(data_dtype(1e-8), y_max - y_min),
            "k": max(data_dtype(1.0), 0.2 * (x_min + x_max)),
            "y0": y_min,
        }

    @classmethod
    def param_bounds(cls) -> ParametersBounds:
        """Boundary values for the `MichaelisMentenSaturation`."""
        return ParametersBounds(
            min={"v_max": neg_inf, "k": data_dtype(0.0), "y0": neg_inf},
            max=dict.fromkeys(["v_max", "k", "y0"], pos_inf),
        )


class Logistic:
    """Logistic (sigmoid) model: f(x) = y₀ + L/(1 + e^(-k(x - x₀))).

    Models S-shaped growth with lower and upper asymptotes.

    References
    ----------
    1. Wikipedia contributors. (2025, November 29). Logistic regression. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:34, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Logistic_regression&oldid=1324697470
    """

    def __call__(
        self,
        x: AnyShapeDataDtypeArray,
        L: data_dtype,  # noqa: N803 L ok as argument name to follow reference
        k: data_dtype,
        x0: data_dtype,
        y0: data_dtype,
    ) -> AnyShapeDataDtypeArray:
        """Evaluate the `Logistic` function at each point in `x`.

        Parameters
        ----------
        x : AnyShapeDataDtypeArray
            Input values
        L : data_dtype
            Maximum value of the curve (carrying capacity)
        k : data_dtype
            Steepness of the curve
        x0 : data_dtype
            Midpoint (inflection point) of the sigmoid
        y0 : data_dtype
            Minimum value (lower asymptote)

        Returns
        -------
        AnyShapeDataDtypeArray
            Logistic function value at each point in `x`.
        """
        return y0 + L / (1.0 + np.exp(-k * (x - x0)))  # type: ignore[reportReturnType] return type is data_dtype alright

    @classmethod
    def param_guess(
        cls, x_min: data_dtype, x_max: data_dtype, y_min: data_dtype, y_max: data_dtype
    ) -> dict[str, data_dtype]:
        """Parameter guesses for a fit initial values.

        x are the function input values, y the predictions in the data points. `L` is typically close to the
        prediction values range, `k` controls the width of the transition interval between the 2 asymptotes (guess is
        0.5), `x0` the midpoint is in the middle of the input values distributions, `y0` the lower asymptote should be
        close to the minimum of the predictions.

        Parameters
        ----------
        x_min : data_dtype
            Minimum input value
        x_max : data_dtype
            Maximum input value
        y_min : data_dtype
            Minimum prediction value
        y_max : data_dtype
            Maximum prediction value

        Returns
        -------
        dict[str, data_dtype]
            Dictionary mapping parameter names to guesses.
        """
        return {
            "L": max(data_dtype(1e-8), y_max - y_min),
            "k": data_dtype(0.5),
            "x0": 0.5 * (x_min + x_max),
            "y0": y_min,
        }

    @classmethod
    def param_bounds(cls) -> ParametersBounds:
        """Boundary values for the `Logistic`."""
        return ParametersBounds(
            min={"L": neg_inf, "k": data_dtype(0.0), "x0": neg_inf, "y0": neg_inf},
            max=dict.fromkeys(["L", "k", "x0", "y0"], pos_inf),
        )


class PeakModel:
    """Peak model (gamma-like): f(x) = a·x·e^(-x/b).

    Models a single peak with exponential decay, useful for
    representing optimal consumption with negative effects beyond peak.

    References
    ----------
    1. Wikipedia contributors. (2025, November 4). Gamma distribution. In Wikipedia, The Free Encyclopedia.
       Retrieved 19:38, December 1, 2025,
       from https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1320436343
    """

    def __call__(self, x: AnyShapeDataDtypeArray, a: data_dtype, b: data_dtype) -> AnyShapeDataDtypeArray:
        """Evaluate `PeakModel` function at each point in `x`.

        Parameters
        ----------
        x : AnyShapeDataDtypeArray
            Input values
        a : data_dtype
            Amplitude scaling factor
        b : data_dtype
            Decay rate parameter

        Returns
        -------
        AnyShapeDataDtypeArray
            `PeakModel` values at each point in `x`
        """
        return a * x * np.exp(-x / np.maximum(b, 1e-9))

    @classmethod
    def param_guess(cls, x_min: data_dtype, x_max: data_dtype, y_max: data_dtype) -> dict[str, data_dtype]:
        """Parameter guesses for a fit initial values.

        x are the function input values, y the predictions in the data points. `a`'s guess is the maximum prediction
        value, `b` guess is the middle point of the input value range.

        Parameters
        ----------
        x_min : data_dtype
            Minimum input value
        x_max : data_dtype
            Maximum input value
        y_max : data_dtype
            Maximum prediction value

        Returns
        -------
        dict[str, data_dtype]
            Dictionary mapping parameter names to guesses.
        """
        return {"a": y_max, "b": max(data_dtype(1.0), 0.5 * (x_min + x_max))}

    @classmethod
    def param_bounds(cls) -> ParametersBounds:
        """Boundary values for the `Logistic`."""
        return ParametersBounds(min={"a": neg_inf, "b": data_dtype(0.0)}, max=dict.fromkeys(["a", "b"], pos_inf))


class Peak2Model:
    """Quadratic peak model: f(x) = a·x²·e^(-x/b).

    Similar to `PeakModel` but with quadratic growth before decay.
    """

    def __call__(self, x: AnyShapeDataDtypeArray, a: data_dtype, b: data_dtype) -> AnyShapeDataDtypeArray:
        """Evaluate `Peak2Model` at each point in `x`.

        Parameters
        ----------
        x : AnyShapeDataDtypeArray
            Input values
        a : data_dtype
            Amplitude scaling factor
        b : data_dtype
            Decay rate parameter

        Returns
        -------
        AnyShapeDataDtypeArray
            `Peak2Model` values at each point in `x`.
        """
        return a * (x**2) * np.exp(-x / np.maximum(b, 1e-9))

    @classmethod
    def param_guess(cls, x_min: data_dtype, x_max: data_dtype, y_max: data_dtype) -> dict[str, data_dtype]:
        """Parameter guesses for a fit initial values.

        x are the function input values, y the predictions in the data points. `a`'s guess is the maximum prediction
        value divided by the maximum input value squared, `b` guess is the middle point of the input value range.

        Parameters
        ----------
        x_min : data_dtype
            Minimum input value
        x_max : data_dtype
            Maximum input value
        y_max : data_dtype
            Maximum prediction value

        Returns
        -------
        dict[str, data_dtype]
            Dictionary mapping parameter names to guesses.
        """
        return {
            "a": max(data_dtype(1e-6), y_max / max(1.0, x_max**2)),
            "b": max(data_dtype(1.0), 0.5 * (x_min + x_max)),
        }

    @classmethod
    def param_bounds(cls) -> ParametersBounds:
        """Boundary values for the `Logistic`."""
        return ParametersBounds(min={"a": neg_inf, "b": data_dtype(0.0)}, max=dict.fromkeys(["a", "b"], pos_inf))
