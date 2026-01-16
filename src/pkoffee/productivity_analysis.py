"""Coffee Productivity analysis module."""

import argparse
import logging

import numpy as np
import pandas as pd

from pkoffee.data import data_dtype, extract_arrays, load_csv
from pkoffee.fit_model import Model, fit_model
from pkoffee.fit_model_io import pkoffee_function_id_mapping, save_models
from pkoffee.parametric_function import Logistic, MichaelisMentenSaturation, Peak2Model, PeakModel, Quadratic


def default_models(x: np.ndarray, y: np.ndarray) -> list[Model]:
    """Generate model configurations with suited initial parameter guesses.

    Parameters
    ----------
    x : np.ndarray
        Input data (cups).
    y : np.ndarray
        Output data (productivity).

    Returns
    -------
    list[Model]
        List of model configurations ready for fitting.
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    return [
        Model(
            name="Quadratic",
            function=Quadratic(),
            params=Quadratic.param_guess(y_min=y_min),
            bounds=Quadratic.param_bounds(),
        ),
        Model(
            name="Michaelis-Menten",
            function=MichaelisMentenSaturation(),
            params=MichaelisMentenSaturation.param_guess(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
            bounds=MichaelisMentenSaturation.param_bounds(),
        ),
        Model(
            name="Logistic",
            function=Logistic(),
            params=Logistic.param_guess(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
            bounds=Logistic.param_bounds(),
        ),
        Model(
            name="Peak",
            function=PeakModel(),
            params=PeakModel.param_guess(x_min=x_min, x_max=x_max, y_max=y_max),
            bounds=PeakModel.param_bounds(),
        ),
        Model(
            name="Peak²",
            function=Peak2Model(),
            params=Peak2Model.param_guess(x_min=x_min, x_max=x_max, y_max=y_max),
            bounds=Peak2Model.param_bounds(),
        ),
    ]


def fit_all_models(
    data: pd.DataFrame,
    max_iterations: int = 20000,
) -> list[Model]:
    r"""Fit all available models to the data and rank by R².

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'cups' and 'productivity' columns.
    max_iterations : int, optional
        Maximum iterations for optimization, by default 20000.

    Returns
    -------
    list[ModelResult]
        List of fitted models, sorted by R² (descending). Models for which fitting failed
        are still in the list with default values (R²=-inf).

    Examples
    --------
    >>> data = load_csv(Path(tmpfile.name))  # doctest: +SKIP
    >>> models = fit_all_models(data)  # doctest: +SKIP
    >>> for model in models:  # doctest: +SKIP
    >>>     print(f"{model.name}: R² = {model.r_squared:.4f}")# doctest: +SKIP
    Quadratic: R² = 0.9978
    Peak²: R² = 0.9115
    Logistic: R² = 0.7525
    Peak: R² = 0.6699
    Michaelis-Menten: R² = 0.2347
    """
    logger = logging.getLogger(__name__)
    x, y = extract_arrays(data)
    models = default_models(x, y)

    fitted_models = []
    for mdl in models:
        logger.info("Fitting model %s", mdl.name)
        try:
            fitted_model, _ = fit_model(x, y, mdl, max_iterations)
            fitted_models.append(fitted_model)
        except (ValueError, RuntimeError):
            logger.warning("Warning: failed to fit %s model.", mdl.name, exc_info=True)
            fitted_models.append(mdl)
        logger.info("Successfully fitted model %s", mdl.name)

    Model.sort(fitted_models)

    return fitted_models


def format_model_rankings(fitted_models: list[Model]) -> str:
    r"""Print a formatted table of model rankings.

    Parameters
    ----------
    fitted_models : list[ModelResult]
        List of fitted models, should be sorted by R².

    Examples
    --------
    >>> from pkoffee.data import load_csv
    >>> from pkoffee.productivity_analysis import fit_all_models
    >>> data = load_csv(Path("coffee_productivity.csv")  # doctest: +SKIP
    >>> models = fit_all_models(data)  # doctest: +SKIP
    >>> print(format_model_rankings(models))  # doctest: +SKIP
    Model Rankings:
    ══════════════════════════════════════════════════
    Rank   Model                R² Score
    ══════════════════════════════════════════════════
    1      Quadratic            0.9978
    2      Peak²                0.9115
    3      Logistic             0.7525
    4      Peak                 0.6699
    5      Michaelis-Menten     0.2347
    ══════════════════════════════════════════════════
    """
    if not fitted_models:
        return ""

    Model.sort(fitted_models)

    rank_nb_char = 6
    model_nb_char = 20
    line_nb_char = 50

    ranking_str = "Model Rankings:\n"
    ranking_str += "═" * line_nb_char + "\n"
    ranking_str += f"{'Rank':<{rank_nb_char}} {'Model':<{model_nb_char}} R² Score\n"
    ranking_str += "═" * line_nb_char + "\n"
    for rank, model in enumerate(fitted_models, start=1):
        r2_str = f"{model.r_squared:.4f}" if np.isfinite(model.r_squared) else "N/A"
        ranking_str += f"{rank:<{rank_nb_char}} {model.name:<{model_nb_char}} {r2_str}\n"
    ranking_str += "═" * 50

    return ranking_str


def analyze(args: argparse.Namespace) -> None:
    """Fit models on input data and save them to file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Load data
    logger = logging.getLogger(__name__)
    logger.info("Loading data from: %s", args.data_file)
    data = load_csv(args.data_file)
    logger.info("Loaded %s data points", len(data))

    # Fit models
    logger.info("Fitting models...")
    models = fit_all_models(data)
    logger.info("Successfully fitted %s models", len([m for m in models if m.r_squared > -data_dtype(np.inf)]))

    # Print rankings if requested
    if args.show_rankings:
        print(format_model_rankings(models))  # noqa: T201 Print is explicitly requested by user

    logger.info("Saving models to %s", args.output)
    save_models(models, pkoffee_function_id_mapping().inv, args.output)

    logger.info("✓ Analysis complete!")
