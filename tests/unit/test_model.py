from pathlib import Path

import joblib
import numpy as np
import pytest

from fashion_mnist_mlops.config import load_config
from fashion_mnist_mlops.model import build_model, load_artifact


def toy_dataset(samples_per_class: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    images = np.zeros((samples_per_class * 2, 28, 28), dtype=np.float32)
    labels = np.repeat([0, 1], samples_per_class)

    images[:samples_per_class, :, 8:12] = 255
    images[samples_per_class:, 8:12, :] = 255

    images += rng.normal(0, 3, size=images.shape)
    return np.clip(images, 0, 255), labels


@pytest.mark.unit
def test_model_fits_and_predicts_probabilities() -> None:
    config = load_config()
    config["MODEL"]["early_stopping"] = "false"
    images, labels = toy_dataset()
    model = build_model(config).fit(images, labels)

    assert np.mean(model.predict(images) == labels) == 1.0
    np.testing.assert_allclose(model.predict_proba(images[:2]).sum(axis=1), 1.0)


@pytest.mark.unit
def test_artifact_contract(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    joblib.dump({"wrong": True}, artifact_path)
    with pytest.raises(ValueError, match="Invalid model artifact"):
        load_artifact(artifact_path)
