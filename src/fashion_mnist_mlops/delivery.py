"""Delivery adapters for model prediction events."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import httpx

from fashion_mnist_mlops.events import DeliveryReceipt, PredictionEvent


class DeliveryError(RuntimeError):
    """Raised when a prediction cannot be delivered reliably."""


class PredictionPublisher(Protocol):
    def publish(self, event: PredictionEvent) -> DeliveryReceipt:
        ...


class DisabledPublisher:
    def publish(self, event: PredictionEvent) -> DeliveryReceipt:
        return DeliveryReceipt(mode="disabled", destination="none")


class HBaseGatewayPublisher:
    """Authenticated client for the internal HBase gateway."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url or not username or not password:
            raise ValueError("HBase gateway URL and credentials are required")

        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            auth=(username, password),
            timeout=timeout_seconds,
            transport=transport
        )

    def publish(self, event: PredictionEvent) -> DeliveryReceipt:
        try:
            response = self._client.post(
                f"{self._base_url}/predictions",
                json=event.model_dump(mode="json")
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise DeliveryError(f"HBase gateway rejected prediction: {e}") from e
        return DeliveryReceipt(mode="hbase", destination=self._base_url)


@lru_cache(maxsize=1)
def get_prediction_publisher() -> PredictionPublisher:
    mode = os.getenv("PREDICTION_DELIVERY", "disabled").strip().lower()
    if mode == "disabled":
        return DisabledPublisher()
    if mode == "hbase":
        return HBaseGatewayPublisher(
            base_url=os.environ["HBASE_GATEWAY_URL"],
            username=os.environ["HBASE_GATEWAY_USERNAME"],
            password=os.environ["HBASE_GATEWAY_PASSWORD"],
            timeout_seconds=float(os.getenv("HBASE_GATEWAY_TIMEOUT_SECONDS", "5")),
        )
    raise ValueError(f"Unsupported PREDICTION_DELIVERY mode: {mode}")
