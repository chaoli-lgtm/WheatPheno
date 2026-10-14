"""Minimal, inference-only compatibility for the trusted historical pickles.

The original estimator was serialized as ``run_validation.RankSelect``.  Loading
that original research script would also execute its unrelated module setup.
This module registers only the exact transformer interface used for prediction.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RankSelect(BaseEstimator, TransformerMixin):
    """Apply the stored feature indices; this distribution does not fit models."""

    def __init__(self, k: int = 8):
        self.k = k

    def fit(self, X, y=None):
        raise RuntimeError(
            "The bundled model is frozen. RankSelect supports inference only."
        )

    def transform(self, X):
        if not hasattr(self, "indices_"):
            raise RuntimeError("The frozen selector has no stored feature indices.")
        return np.asarray(X)[:, self.indices_]


def register_legacy_selector() -> None:
    """Install the lightweight pickle namespace without importing old scripts.

    Call this only after the bundled artifact has passed its trusted SHA256
    check. It deliberately replaces any existing research-script namespace.
    """
    module = ModuleType("run_validation")
    module.__doc__ = "WheatPheno inference-only pickle compatibility namespace."
    module.RankSelect = RankSelect
    sys.modules["run_validation"] = module
