"""FastAPI inference service for the trained Fashion-MNIST model."""

from __future__ import annotations

import os
from functools import lru_cache
from hashlib import sha256

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from fashion_mnist_mlops.config import CLASS_NAMES, PROJECT_ROOT
from fashion_mnist_mlops.delivery import DeliveryError, get_prediction_publisher
from fashion_mnist_mlops.events import PredictionEvent
from fashion_mnist_mlops.model import load_artifact

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/model.joblib"


class PredictionRequest(BaseModel):
    pixels: list[float] = Field(
        min_length=28 * 28,
        max_length=28 * 28,
        description="Flattened 28x28 grayscale image.",
    )

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, pixels: list[float]) -> list[float]:
        array = np.asarray(pixels, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("pixels must contain finite values")
        if array.min() < 0 or array.max() > 255:
            raise ValueError("pixel values must be in [0, 255]")
        return pixels


class PredictionResponse(BaseModel):
    prediction_id: str
    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    delivery: str


@lru_cache
def get_artifact(model_path: str) -> dict:
    return load_artifact(model_path)


def configured_model_path() -> str:
    return os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fashion-MNIST classifier",
        version="0.1.0",
        description="LogRegresion inference server",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        path = configured_model_path()
        try:
            get_artifact(path)
        except (FileNotFoundError, ValueError, OSError) as e:
            raise HTTPException(status_code=503, detail=f"Model is unavailable: {e}") from e
        return {"status": "ok"}

    @app.get("/model")
    def model_info() -> dict:
        return get_artifact(configured_model_path())["metadata"]

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        artifact = get_artifact(configured_model_path())
        model = artifact["model"]
        image = np.asarray(request.pixels, dtype=np.float32).reshape(1, 28, 28)
        probabilities = model.predict_proba(image)[0]
        class_id = int(np.argmax(probabilities))
        names = artifact["metadata"].get("classes", list(CLASS_NAMES))

        event = PredictionEvent(
            request_sha256=sha256(image.tobytes()).hexdigest(),
            class_id=class_id,
            label=names[class_id],
            confidence=float(probabilities[class_id]),
            probabilities={
                name: float(value) for name, value in zip(names, probabilities, strict=True)
            },
            model_config_sha256=artifact["metadata"].get("config_sha256"),
        )

        try:
            receipt = get_prediction_publisher().publish(event)
        except (DeliveryError, KeyError, ValueError) as e:
            raise HTTPException(
                status_code=503,
                detail=f"Prediction delivery failed: {e}",
            ) from e

        return PredictionResponse(
            prediction_id=event.prediction_id,
            class_id=class_id,
            label=names[class_id],
            confidence=float(probabilities[class_id]),
            probabilities=event.probabilities,
            delivery=receipt.mode,
        )

    return app


app = create_app()
