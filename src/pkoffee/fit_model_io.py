"""Input/Output for models.

A model's `ParametricFunction` is not directly saved, only an identifier string is written to file. In order to
reconstruct the model from the file, the same mapping from function to identifier needs to be available. This module
implements a function returning a bidirectional mapping for the `ParametricFunction` implemented in the `pkoffeee`
package. This mapping can be extended with additional functions to save other models.
"""

import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path

import tomlkit
from bidict import bidict
from tomlkit import aot, document, item

from pkoffee.fit_model import Model
from pkoffee.parametric_function import (
    Logistic,
    MichaelisMentenSaturation,
    Peak2Model,
    PeakModel,
    Quadratic,
)


def pkoffee_function_id_mapping() -> bidict:
    """Bidirectional mapping from string Identifiers to the ParametricFunctions implemented in `pkoffee`."""
    return bidict(
        {
            "Quadratic": Quadratic,
            "MichaelisMentenSaturation": MichaelisMentenSaturation,
            "Logistic": Logistic,
            "PeakModel": PeakModel,
            "Peak2Model": Peak2Model,
        }
    )


class UnsupportedModelFormatError(NotImplementedError):
    """Exception for non-implemented model file format."""

    def __init__(self, file_format: str) -> None:
        super().__init__(f"Model format {file_format} not supported. See ModelFileFormat.")


class ModelFileFormat(StrEnum):
    """Available format for saving models to file."""

    TOML = "toml"
    JSON = "json"


def file_format_from_path(file_path: Path) -> ModelFileFormat:
    """Determine models's file format from a file path extension.

    Parameters
    ----------
    file_path : Path
        Path to a file, eg "model.toml"

    Returns
    -------
    ModelFileFormat
        File format

    Raises
    ------
    UnsupportedModelFormatError
        If the file format is not supported
    """
    try:
        return ModelFileFormat(file_path.suffix[1:])  # remove '.' from suffix
    except KeyError as e:
        raise UnsupportedModelFormatError(file_path.suffix) from e


def save_models_json(model_dicts: Iterable[dict], output_path: Path) -> None:
    """Save the model dictionary representation to a json file.

    Parameters
    ----------
    model_dicts : Iterable[dict]
        Models dictionary representation
    output_path : Path
        Path to save the models
    """
    with output_path.open("w") as of:
        of.write(json.dumps({"Models": model_dicts}))


def save_models_toml(model_dicts: Iterable[dict], output_path: Path) -> None:
    """Save the model dictionaries representation to a toml file.

    Parameters
    ----------
    model_dicts : Iterable[dict]
        Models dictionary representation
    output_path : Path
        Path to save the models
    """
    toml_doc = document()
    models_array = aot()
    for md in model_dicts:
        models_array.append(item(md))
    toml_doc.append("Models", models_array)
    with output_path.open("w") as of:
        of.write(toml_doc.as_string())


def save_models(
    models: Iterable[Model], function_to_str: Mapping, output_path: Path, file_format: ModelFileFormat | None = None
) -> None:
    """Save the models to disk.

    Parameters
    ----------
    models : Iterable[Model]
        Collection of models to save
    function_to_str : Mapping
        Mapping of function to string identifier used as function representation in the model's file.
    output_path : Path
        Path to the model's file.
    file_format : ModelFileFormat
        The format of the model's file
    """
    if file_format is None:
        file_format = file_format_from_path(output_path)
    model_dicts = [m.to_dict(function_to_str) for m in models]
    match file_format:
        case ModelFileFormat.JSON:
            save_models_json(model_dicts, output_path)
        case ModelFileFormat.TOML:
            save_models_toml(model_dicts, output_path)
        case _:
            raise UnsupportedModelFormatError(str(file_format))


def load_models_json(model_file: Path, str_to_function: Mapping) -> list[Model]:
    """Load models from json file.

    Parameters
    ----------
    model_file : Path
        Path to the models' file
    str_to_function : Mapping
        Mapping of function string identifier to function classes

    Returns
    -------
    list[Model]
        Loaded models
    """
    with model_file.open("r") as mdlf:
        models_dict = json.loads(mdlf.read())["Models"]
    return [Model.from_dict(m_d, str_to_function) for m_d in models_dict]


def load_models_toml(model_file: Path, str_to_function: Mapping) -> list[Model]:
    """Load models from toml file.

    Parameters
    ----------
    model_file : Path
        Path to the models' file
    str_to_function : Mapping
        Mapping of function string identifier to function classes

    Returns
    -------
    list[Model]
        Loaded models
    """
    with model_file.open("r") as mdlf:
        models_dict = tomlkit.parse(mdlf.read())["Models"]
    return [Model.from_dict(m_d, str_to_function) for m_d in models_dict]  # pyright: ignore[reportGeneralTypeIssues] models_dict is iterable alright


def load_models(
    model_file: Path, str_to_function: Mapping, file_format: ModelFileFormat | None = None
) -> list[Model]:
    """Load models from file.

    Parameters
    ----------
    model_file : Path
        Path to the model's file
    str_to_function : Mapping
        Mapping of function string identifier to function classes
    file_format : ModelFileFormat
        Format of the model file

    Returns
    -------
    list[Model]
        Loaded models
    """
    if file_format is None:
        file_format = file_format_from_path(model_file)
    match file_format:
        case ModelFileFormat.JSON:
            return load_models_json(model_file, str_to_function)
        case ModelFileFormat.TOML:
            return load_models_toml(model_file, str_to_function)
        case _:
            raise UnsupportedModelFormatError(str(file_format))
