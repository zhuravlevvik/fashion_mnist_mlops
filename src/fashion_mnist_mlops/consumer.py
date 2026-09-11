"""Kafka Consumer that persists prediction events through the HBase gateway."""

from __future__ import annotations

import logging
import os
import signal
from typing import Any

from confluent_kafka import Consumer, KafkaException

from fashion_mnist_mlops.delivery import PredictionPublisher, build_hbase_gateway_publisher
from fashion_mnist_mlops.events import PredictionEvent

LOGGER = logging.getLogger(__name__)


class PredictionConsumer:
    """At-least-once consumer: offsets are committed only after HBase acknowledges."""

    def __init__(self, consumer: Any, sink: PredictionPublisher, topic: str) -> None:
        self._consumer = consumer
        self._sink = sink
        self._consumer.subscribe([topic])

    def run_once(self, timeout_seconds: float = 1.0) -> bool:
        message = self._consumer.poll(timeout_seconds)
        if message is None:
            return False
        if message.error():
            raise KafkaException(message.error())

        event = PredictionEvent.model_validate_json(message.value())
        self._sink.publish(event)
        self._consumer.commit(message=message, asynchronous=False)
        return True

    def close(self) -> None:
        self._consumer.close()


def build_consumer() -> PredictionConsumer:
    bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    topic = os.getenv("KAFKA_PREDICTIONS_TOPIC", "fashion.predictions.v1")
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": os.getenv("KAFKA_CONSUMER_GROUP", "fashion-hbase-writer-v1"),
            "client.id": "fashion-mnist-hbase-consumer",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    return PredictionConsumer(consumer, build_hbase_gateway_publisher(), topic)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    running = True

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    consumer = build_consumer()

    LOGGER.info("Kafka prediction consumer started")

    try:
        while running:
            consumer.run_once()
    finally:
        consumer.close()
        LOGGER.info("Kafka prediction consumer stopped")


if __name__ == "__main__":
    main()
