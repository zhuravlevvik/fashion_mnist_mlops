"""HBase repository implemented through the Thrift gateway."""


from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import happybase

from fashion_mnist_mlops.events import PredictionEvent

ConnectionFactory = Callable[[], Any]


class HBaseRepository:
    """Stores one JSON payload per row while keeping useful columns queryable."""

    def __init__(
        self,
        host: str,
        port: int = 9090,
        table_name: str = "fashion_predictions",
        timeout_ms: int = 5000,
        connection_factory: ConnectionFactory | None = None
    ) -> None:
        self.host = host
        self.port = port
        self.table_name = table_name
        self.timeout_ms = timeout_ms
        self._connection_factory = connection_factory or self._new_connection

    @classmethod
    def from_env(cls) -> HBaseRepository:
        return cls(
            host=os.getenv("HBASE_HOST", "hbase"),
            port=int(os.getenv("HBASE_THRIFT_PORT", "9090")),
            table_name=os.getenv("HBASE_TABLE", "fashion_predictions"),
            timeout_ms=int(os.getenv("HBASE_TIMEOUT_MS", "5000")),
        )

    def _new_connection(self) -> happybase.Connection:
        return happybase.Connection(
            host=self.host,
            port=self.port,
            timeout=self.timeout_ms,
            autoconnect=False,
        )

    def _with_connection(self, operation: Callable[[Any], Any]) -> Any:
        connection = self._connection_factory()

        try:
            connection.open()
            return operation(connection)
        finally:
            connection.close()

    def ensure_schema(self) -> None:
        table_name = self.table_name.encode()

        def create_if_missing(connection: Any) -> None:
            if table_name not in connection.tables():
                connection.create_table(self.table_name, {"p": {"max_versions": 1}})

        self._with_connection(create_if_missing)

    def ping(self) -> None:
        self._with_connection(lambda connection: connection.tables())

    def put(self, event: PredictionEvent) -> None:
        payload = json.dumps(event.model_dump(mode="json"), sort_keys=True).encode()
        columns = {
            b"p:payload": payload,
            b"p:occured_at": event.occured_at.encode(),
            b"p:label": event.label.encode(),
            b"p:confidence": format(event.confidence, ".12g").encode()
        }

        def write(connection: Any) -> None:
            connection.table(self.table_name).put(event.prediction_id.encode(), columns)

        self._with_connection(write)

    def get(self, prediction_id: str) -> PredictionEvent | None:
        def read(connection: Any) -> dict[bytes, bytes]:
            return connection.table(self.table_name).row(prediction_id.encode())

        row = self._with_connection(read)
        if not row:
            return None
        return PredictionEvent.model_validate_json(row[b"p:payload"])
