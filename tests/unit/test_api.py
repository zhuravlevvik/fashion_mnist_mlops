from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from fashion_mnist_mlops.api import create_app, get_artifact
from fashion_mnist_mlops.config import load_config
from fashion_mnist_mlops.model import build_model
from tests.unit.test_model import toy_dataset


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    config = load_config()
    config["MODEL"]["early_stopping"] = "false"
    images, labels = toy_dataset()
    model = build_model(config).fit(images, labels)
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "metadata": {"classes": ["vertical", "horizontal"]}}, path)
    return path


@pytest.mark.unit
def test_health_and_prediction(model_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    get_artifact.cache_clear()
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    pixels = [0.0] * (28 * 28)
    for row in range(28):
        for column in range(8, 12):
            pixels[row * 28 + column] = 255.0
    response = client.post("/predict", json={"pixels": pixels})

    assert response.status_code == 200
    assert response.json()["label"] == "vertical"
    assert 0 <= response.json()["confidence"] <= 1


@pytest.mark.unit
def test_prediction_validation(model_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    get_artifact.cache_clear()
    client = TestClient(create_app())

    assert client.post("/predict", json={"pixels": [0]}).status_code == 422
    assert client.post("/predict", json={"pixels": [300] * (28 * 28)}).status_code == 422
