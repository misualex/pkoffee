"""Visualization utilities for coffee productivity analysis."""

import argparse
import logging
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from pkoffee.data import RequiredColumn, data_dtype, load_csv
from pkoffee.fit_model import Model
from pkoffee.fit_model_io import load_models, pkoffee_function_id_mapping


class Show(Enum):
    """To show or not to show a figure."""

    YES = auto()
    NO = auto()


class NoModelProvidedError(ValueError):
    """Exception for data input failure."""

    def __init__(self) -> None:
        super().__init__("No Model provided")


class FigureParameters(NamedTuple):
    """Usual parameters of `matlplotlib.figure.Figure`.

    Attributes
    ----------
    y_limits : tuple[float, float] | None
        Limits of the y axis (min, max), default is None (to let matplotlib determine the values).
    figsize: tuple[float, float] | None
        Figure size in inches (matplotlib unit...) as (width, height). Default is (12, 7)
    dpi: int
        Drop Per Inch (number of ink droplets per inch) to use for the figure. Default is 150.
    """

    y_limits: tuple[float, float] | None = None
    figsize: tuple[float, float] | None = (12, 7)
    dpi: int = 150


def draw_data_violin(ax: Axes, data: pd.DataFrame) -> None:
    """Draw a violin plot of the data on `ax`.

    Parameters
    ----------
    ax : Axes
        Axes onto which to draw
    data : pd.DataFrame
        The DataFrame with the data to draw
    """
    # Set seaborn style for beautiful plots
    sns.set_theme(style="whitegrid", palette="husl")
    # Create violin plot showing distribution by cups
    sns.violinplot(
        data=data,
        x=RequiredColumn.CUPS,
        y=RequiredColumn.PRODUCTIVITY,
        hue=RequiredColumn.CUPS,
        ax=ax,
        inner="quartile",
        cut=0,
        density_norm="width",
        palette="Greens",
        linewidth=0.8,
        legend=False,
        alpha=0.7,
    )


def draw_model_lines(
    ax: Axes, x_smooth: np.ndarray, y_smooth: list[np.ndarray | None], labels: list[str], fig_params: FigureParameters
) -> None:
    """Draw the models prediction lines onto `ax`.

    Parameters
    ----------
    ax : Axes
        Axe onto which to draw
    x_smooth : np.ndarray
        x values of the line points
    y_smooth : list[np.ndarray  |  None]
        List of y values of the line points, one per element in `labels`
    labels : list[str]
        List of labels to use in the plot legend
    fig_params : FigureParameters
        Figure parameters
    """
    # Plot the curves
    legend_elements = []
    for idx, (mod_y_draw, mod_label, color) in enumerate(
        zip(y_smooth, labels, colormaps["tab10"](np.linspace(0, 1, len(y_smooth))), strict=True)
    ):
        # If the predictions are not None: draw them
        # In any case: add the model to the legend so its label shows in the plot.
        if mod_y_draw is not None:
            ax.plot(
                x_smooth,
                mod_y_draw,
                linewidth=2.5,
                color=color,
                linestyle="solid",
                alpha=0.9,
                zorder=10 + idx,
            )
        legend_elements.append(Line2D([0], [0], color=color, linestyle="solid", linewidth=2.5, label=mod_label))

    # Styling
    ax.set_title(
        "Coffee Consumption vs Productivity: Distribution and Model Fits",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Coffee Cups Consumed", fontsize=12, fontweight="bold")
    ax.set_ylabel("Productivity", fontsize=12, fontweight="bold")
    ax.set_ylim(fig_params.y_limits)
    ax.grid(True, alpha=0.3, linestyle="--")  # noqa: FBT003 ax.grid boolean value is external
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=True,
        shadow=True,
        fancybox=True,
        fontsize=10,
        title="Model Rankings",
        title_fontsize=11,
    )
    plt.tight_layout()


def plot_models(
    data: pd.DataFrame,
    fitted_models: list[Model],
    output_path: Path | None = None,
    fig_params: FigureParameters | None = None,
    show: Show = Show.YES,
) -> None:
    r"""Create a comprehensive analysis plot with data distribution and model fits.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'cups' and 'productivity' columns.
    fitted_models : list[ModelResult]
        List of fitted models to overlay on the plot.
    fig_params : FigureParameters | None
        Figure parameters
    output_path : Path | None
        Path to save the figure. If None, figure is not saved.
    show : Show
        Whether to display the plot, by default YES.

    Returns
    -------
    plt.Figure
        The created matplotlib figure.

    Examples
    --------
    >>> from pkoffee.data import load_csv
    >>> from pkoffee.productivity_analysis import fit_all_models
    >>> data = load_csv(Path("coffee_productivity.csv"))  # doctest: +SKIP
    >>> models = fit_all_models(data)  # doctest: +SKIP
    >>> plot_models(data, models, Path("analysis.png"))  # doctest: +SKIP
    """
    if fig_params is None:
        fig_params = FigureParameters(figsize=(12, 7), dpi=150, y_limits=(-0.2, 8.0))

    fig, ax = plt.subplots(figsize=fig_params.figsize)

    # Prepare smooth x values for plotting fitted curves
    x_min = data[RequiredColumn.CUPS].min()
    x_max = data[RequiredColumn.CUPS].max()
    x_smooth = np.linspace(x_min, x_max, 300)

    y_smooth = []
    for mdl in fitted_models:
        if mdl.r_squared > -np.inf:
            y_smooth.append(mdl.predict(x_smooth))
        else:
            y_smooth.append(None)

    draw_data_violin(ax, data)
    draw_model_lines(
        ax,
        x_smooth,
        y_smooth,
        [f"{m.name}: (R² = {m.r_squared:.3f})" for m in fitted_models],
        fig_params,
    )

    # Save figure if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=fig_params.dpi, bbox_inches="tight")
        logger = logging.getLogger(__name__)
        logger.info("Models plot saved to: %s", output_path)

    # Display figure if requested
    if show == Show.YES:
        plt.show()


def create_comparison_plot(
    data: pd.DataFrame,
    fitted_models: list[Model],
    output_path: Path | None = None,
    fig_params: FigureParameters | None = None,
    show: Show = Show.NO,
) -> None:
    """Create a multi-panel comparison plot showing each model separately.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'cups' and 'productivity' columns.
    fitted_models : list[ModelResult]
        List of fitted models to display.
    output_path : str or Path, optional
        Path to save the figure.
    fig_params : FigureParameters
        Configuration value for matlplotlib figure
    show : Show
        Whether to show the figure or not.
    """
    if fig_params is None:
        fig_params = FigureParameters(figsize=(14, 10), dpi=150)
    if not fitted_models:
        raise NoModelProvidedError

    # Calculate grid dimensions
    n_models = len(fitted_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_params.figsize, squeeze=False)
    axes = axes.flatten()

    cups = data[RequiredColumn.CUPS].to_numpy(dtype=data_dtype)
    productivity = data[RequiredColumn.PRODUCTIVITY].to_numpy(dtype=data_dtype)
    x_smooth = np.linspace(cups.min(), cups.max(), 300)

    for model, ax in zip(fitted_models, axes, strict=False):  # may be more axes than models
        # Scatter plot of data
        ax.scatter(cups, productivity, alpha=0.3, s=10, color="gray", label="Data")

        # Model fit
        y_smooth = model.predict(x_smooth)
        ax.plot(x_smooth, y_smooth, lw=2, color="blue", label="Model Fit")

        # Styling
        ax.set_xlabel("Cups", fontsize=10)
        ax.set_ylabel("Productivity", fontsize=10)
        ax.set_title(f"{model.name}\nR² = {model.r_squared:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)  # noqa: FBT003
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Model Comparison: Individual Fits",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=fig_params.dpi, bbox_inches="tight")
        logger = logging.getLogger(__name__)
        logger.info("Comparison plot saved to: %s", output_path)

    # Display figure if requested
    if show == Show.YES:
        plt.show()


def visualize(args: argparse.Namespace) -> None:
    """Plot model predictions and data.

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

    # Load models
    logger.info("Loading models from: %s", args.model_file)
    models = load_models(args.model_file, pkoffee_function_id_mapping())
    logger.info("Loaded %s models", len(models))

    # Determine y-axis limits
    y_min = args.y_min if args.y_min is not None else data[RequiredColumn.PRODUCTIVITY].min()
    y_max = args.y_max if args.y_max is not None else data[RequiredColumn.PRODUCTIVITY].max()
    y_limits = (y_min, y_max)

    # Create main analysis plot
    logger.info("Creating analysis plot: %s", args.output)
    plot_models(
        data,
        models,
        output_path=args.output,
        fig_params=FigureParameters(y_limits=y_limits, dpi=args.dpi),
        show=Show.YES if args.no_show else Show.NO,
    )

    # Create comparison plot if requested
    if args.comparison is not None:
        logger.info("Creating comparison plot: %s", args.comparison)
        create_comparison_plot(
            data,
            models,
            output_path=args.comparison,
            fig_params=FigureParameters(dpi=args.dpi),
            show=Show.YES if args.no_show else Show.NO,
        )
