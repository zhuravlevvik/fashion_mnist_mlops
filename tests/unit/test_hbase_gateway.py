from typing import Any

import pytest
from fastapi.testclient import TestClient

from fashion_mnist_mlops.events import PredictionEvent
from fashion_mnist_mlops.hbase_gateway import create_gateway_app


class FakeRepository:
    def __init__(self) -> None:
        self.initialized = False
        self.events: dict[str, PredictionEvent] = {}

    def ensure_schema(self) -> None:
        self.initialized = True

    def ping(self) -> None:
        return None

    def put(self, event: PredictionEvent) -> None:
        self.events[event.prediction_id] = event

    def get(self, prediction_id: str) -> PredictionEvent | None:
        return self.events.get(prediction_id)


@pytest.mark.unit
def test_gateway_authentication_and_round_trip() -> None:
    repository = FakeRepository()
    app = create_gateway_app(
        repository=repository, credentials_provider=lambda: ("fashion-api", "secret")
    )
    payload: dict[str, Any] = {
        "prediction_id": "prediction-19",
        "occured_at": "2026-08-26T10:00:00+00:00",
        "request_sha256": "c" * 64,
        "class_id": 1,
        "label": "Shirt",
        "confidence": 0.9,
        "probabilities": {"Shirt": 0.9},
    }

    with TestClient(app) as client:
        assert repository.initialized is True
        assert client.get("/health").status_code == 200
        assert client.post("/predictions", json=payload).status_code == 401

        created = client.post(
            "/predictions",
            json=payload,
            auth=("fashion-api", "secret"),
        )
        loaded = client.get(
            "/predictions/prediction-19",
            auth=("fashion-api", "secret"),
        )

    assert created.status_code == 201
    assert loaded.status_code == 200
    assert loaded.json()["label"] == "Shirt"
