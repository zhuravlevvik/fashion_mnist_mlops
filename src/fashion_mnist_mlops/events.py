"""Versioned contracts shared by the API, Kafka and HBase gateway."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class PredictionEvent(BaseModel):
    """Immutable result of one model invocation."""

    schema_version: Literal[1] = 1
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    request_sha256: str = Field(min_length=64, max_length=64)
    class_id: int = Field(ge=0, le=9)
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_config_sha256: str | None = None


class DeliveryReceipt(BaseModel):
    """Acknowledgement returned by a prediction delivery adapter."""

    mode: str
    destination: str
