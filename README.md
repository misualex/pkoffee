# PKoffee - Coffee Productivity Analysis

A Conda package implemented in python for analyzing the relationship between coffee consumption and productivity through statistical modeling and visualization.

Project inspired by _Le Café - Oldelaf_ for the S3 School 2026.

[![Le Café - Oldelaf - on YouTube](http://img.youtube.com/vi/UGtKGX8B9hU/0.jpg)](http://www.youtube.com/watch?v=UGtKGX8B9hU "Le Café - Oldelaf")


## Features

- **Data Analysis**: Load and analyze coffee consumption vs productivity datasets
- **Multiple Models**: Fit various mathematical models (quadratic, Michaelis-Menten, logistic, peak functions)
- **Visualization**: Beautiful violin plots with model overlays using Seaborn
- **Model Comparison**: Automatic R² scoring and ranking of models
- **Type-Safe**: Full type hints for better IDE support and code quality

## Installation

```bash
pixi install
```

## Quick Start

```python
from pathlib import Path
from pkoffee.data import load_csv
from pkoffee.productivity_analysis import fit_all_models
from pkoffee.visualization import plot_models

# Load your data
data = load_csv(Path("analysis/coffee_productivity.csv"))

# Fit models
fitted_models = fit_all_models(data)

# Create visualization
plot_models(data, fitted_models, output_path=Path("analysis/results.png"))
```

## Command Line Interface

```bash
# Analyze
cd analysis && pixi shell
# fit models and save to toml
pkoffee analyze --data-file coffee_productivity.csv --output fitted_models.toml --show-rankings
# plot models predictions
pkoffee plot --data-file coffee_productivity.csv --models fitted_models.toml --output analysis.png --y-max 7
```

## Project Structure

```
pkoffee-sol/
├── src/pkoffee/
│   ├── __init__.py               # Python package
│   ├── cli.py                    # Command line interface
│   ├── data.py                   # Data loading utilities
│   ├── fit_model_io.py           # Model writing/reading to file
│   ├── fit_model.py              # Model definition and fitting
│   ├── log.py                    # Logging utilities
│   ├── metrics.py                # Model evaluation metrics
│   ├── parametric_function.py    # Mathematical models as parametric functions
│   ├── productivity_analysis.py  # Coffee analysis
│   └── visualization.py          # Plotting functions
├── README.md
├── pixi.toml
├── pixi.lock
└── pyproject.toml
```

## Models Implemented

1. **Quadratic**: `f(x) = a₀ + a₁x + a₂x²`
2. **Michaelis-Menten (Saturating)**: `f(x) = y₀ + Vₘₐₓ·x/(K + x)`
3. **Logistic**: `f(x) = y₀ + L/(1 + e^(-k(x - x₀)))`
4. **Peak**: `f(x) = a·x·e^(-x/b)`
5. **Peak2**: `f(x) = a·x²·e^(-x/b)`

## Development

```bash
# Activate the environment
pixi shell
# use development dependencies like jupyter notebooks/ipython
jupyterlab
```

## License

MIT License
