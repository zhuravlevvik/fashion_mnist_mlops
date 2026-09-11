import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from fashion_mnist_mlops.consumer import PredictionConsumer
from fashion_mnist_mlops.delivery import DeliveryError, KafkaPredictionPublisher
from fashion_mnist_mlops.events import DeliveryReceipt, PredictionEvent


def prediction_event() -> PredictionEvent:
    return PredictionEvent(
        prediction_id="prediction-kafka-19",
        occurred_at="2026-09-09T10:00:00+01:00",
        request_sha256="ab" * 32,
        class_id=4,
        label="Shirt",
        confidence=0.19,
        probabilities={"Shirt": 0.19},
    )


class FakeProducer:
    def __init__(self, delivery_error: Any = None, remaining: int = 0) -> None:
        self.delivery_error = delivery_error
        self.remaining = remaining
        self.message: dict[str, Any] = {}

    def produce(self, topic: str, **kwargs: Any) -> None:
        self.message = {"topic": topic, **kwargs}
        kwargs["on_delivery"](self.delivery_error, SimpleNamespace())

    def flush(self, _timeout: float) -> int:
        return self.remaining


class FakeMessage:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def error(self):
        return None

    def value(self) -> bytes:
        return self._value


class FakeConsumer:
    def __init__(self, message: FakeMessage | None) -> None:
        self.message = message
        self.topics: list[str] = []
        self.committed = False
        self.closed = False

    def subscribe(self, topics: list[str]) -> None:
        self.topics = topics

    def poll(self, _timeout: float):
        message, self.message = self.message, None
        return message

    def commit(self, **_kwargs: Any) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


class FakeSink:
    def __init__(self) -> None:
        self.events: list[PredictionEvent] = []

    def publish(self, event: PredictionEvent) -> DeliveryReceipt:
        self.events.append(event)
        return DeliveryReceipt(mode="hbase", destination="gateway")


@pytest.mark.unit
def test_kafka_publisher_uses_prediction_id_as_key() -> None:
    producer = FakeProducer()
    publisher = KafkaPredictionPublisher("kafka:9092", "predictions", producer=producer)

    receipt = publisher.publish(prediction_event())

    assert receipt.mode == "kafka"
    assert producer.message["key"] == b"prediction-kafka-19"
    assert producer.message["headers"] == {"schema-version": "1"}


@pytest.mark.unit
def test_kafka_publisher_propagates_delivery_error() -> None:
    publisher = KafkaPredictionPublisher(
        "kafka:9092",
        "predictions",
        producer=FakeProducer(delivery_error="broker unavailable"),
    )

    with pytest.raises(DeliveryError, match="broker unavailable"):
        publisher.publish(prediction_event())


@pytest.mark.unit
def test_kafka_publisher_rejects_flush_timeout() -> None:
    publisher = KafkaPredictionPublisher(
        "kafka:9092", "predictions", producer=FakeProducer(remaining=1)
    )

    with pytest.raises(DeliveryError, match="still queued"):
        publisher.publish(prediction_event())


@pytest.mark.unit
def test_consumer_commits_only_after_hbase_acknowledges() -> None:
    event = prediction_event()
    kafka = FakeConsumer(FakeMessage(event.model_dump_json().encode()))
    sink = FakeSink()
    consumer = PredictionConsumer(kafka, sink, "predictions")

    assert consumer.run_once() is True
    consumer.close()

    assert kafka.topics == ["predictions"]
    assert sink.events == [event]
    assert kafka.committed is True
    assert kafka.closed is True


@pytest.mark.unit
def test_consumer_does_not_commit_invalid_event() -> None:
    event = prediction_event().model_dump(mode="json")
    event["schema_version"] = 2
    kafka = FakeConsumer(FakeMessage(json.dumps(event).encode()))
    consumer = PredictionConsumer(kafka, FakeSink(), "predictions")

    with pytest.raises(ValidationError):
        consumer.run_once()

    assert kafka.committed is False


@pytest.mark.unit
def test_consumer_handles_empty_poll() -> None:
    kafka = FakeConsumer(None)
    consumer = PredictionConsumer(kafka, FakeSink(), "predictions")

    assert consumer.run_once() is False
    assert kafka.committed is False
