"""Versioned feature definitions and package asset discovery."""

from __future__ import annotations

import json
from pathlib import Path
import sys

FROZEN_CONTRACT = "wheatpheno-frozen-ndre-v2"
STANDARD_CONTRACT = "wheatpheno-standard-v1"


def _asset_directory() -> Path:
    """Find fixed packaged assets; never accept user-supplied model paths."""
    package = Path(__file__).resolve().parent
    candidates = (
        package / "_assets" / "models",
        package.parents[1] / "models",  # source/editable installation
        Path(sys.prefix) / "share" / "wheatpheno" / "models",  # wheel data-files
    )
    for directory in candidates:
        if (directory / "schema.json").is_file() and (
            directory / "manifest.json"
        ).is_file():
            return directory
    raise FileNotFoundError(
        "The WheatPheno model assets are missing. Install the complete distribution "
        "or use an editable installation from the project root."
    )


def feature_schema() -> dict:
    """Return a fresh copy of the documented, versioned input contract."""
    schema = json.loads((_asset_directory() / "schema.json").read_text("utf-8"))
    names = schema.get("feature_names", [])
    if len(names) != 72 or len(set(names)) != 72:
        raise ValueError("The bundled feature schema must contain 72 unique inputs.")
    return schema


def feature_names() -> list[str]:
    """Return the exact ordered 72-feature input names for both frozen models."""
    return list(feature_schema()["feature_names"])
