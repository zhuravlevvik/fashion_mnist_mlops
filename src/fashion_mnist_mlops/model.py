"""Model construction and artifact loading."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def flatten_and_scale(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    return images.reshape(images.shape[0], -1) / 255.0


def build_model(config: ConfigParser) -> Pipeline:
    """Build the classical SGD classification pipeline."""

    model = config["MODEL"]
    return Pipeline(
        steps=[
            ("pixels", FunctionTransformer(flatten_and_scale, validate=False)),
            (
                "classifier",
                SGDClassifier(
                    loss=model.get("loss"),
                    penalty=model.get("penalty"),
                    alpha=model.getfloat("alpha"),
                    max_iter=model.getint("max_iter"),
                    early_stopping=model.getboolean("early_stopping"),
                    validation_fraction=model.getfloat("validation_fraction"),
                    n_iter_no_change=model.getint("n_iter_no_change"),
                    random_state=model.getint("random_seed"),
                ),
            ),
        ]
    )


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a model bundle and validate the minimal serving contract."""

    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or "model" not in artifact or "metadata" not in artifact:
        raise ValueError(f"Invalid model artifact: {path}")
    return artifact
