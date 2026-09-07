from typing import Any

import pytest

from fashion_mnist_mlops.events import PredictionEvent
from fashion_mnist_mlops.hbase_repository import HBaseRepository


class FakeTable:
    def __init__(self, rows: dict[bytes, bytes]) -> None:
        self.rows = rows

    def put(self, key: bytes, columns: dict[bytes, bytes]) -> None:
        self.rows[key] = columns

    def row(self, key: bytes) -> dict[bytes, bytes]:
        return self.rows.get(key, {})


class FakeConnection:
    def __init__(self) -> None:
        self.opened = False
        self.created: dict[bytes, dict[str, Any]] = {}
        self.rows: dict[bytes, dict[bytes, bytes]] = {}

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def tables(self) -> list[bytes]:
        return list(self.created)

    def create_table(self, name: str, families: dict[str, Any]) -> None:
        self.created[name.encode()] = families

    def table(self, _name: str) -> FakeTable:
        return FakeTable(self.rows)


@pytest.mark.unit
def test_repository_initializes_and_round_trips_event() -> None:
    connection = FakeConnection()
    repository = HBaseRepository(host="hbase", connection_factory=lambda: connection)
    prediction = PredictionEvent(
        prediction_id="prediction-19",
        occured_at="2026-08-26T10:00:00+08:00",
        request_sha256="ba" * 32,
        class_id=6,
        label="Shirt",
        confidence=0.72,
        probabilities={"Shirt": 0.72},
    )

    repository.ensure_schema()
    repository.ensure_schema()

    repository.put(prediction)

    assert list(connection.created) == [b"fashion_predictions"]
    assert repository.get("prediction-19") == prediction
    assert repository.get("missing") is None
    assert connection.opened is False
