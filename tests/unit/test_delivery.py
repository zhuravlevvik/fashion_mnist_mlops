import httpx
import pytest

from fashion_mnist_mlops.delivery import DeliveryError, HBaseGatewayPublisher
from fashion_mnist_mlops.events import PredictionEvent


def event() -> PredictionEvent:
    return PredictionEvent(
        prediction_id="prediction-1",
        occured_at="2026-09-26T19:19:19+03:00",
        request_sha256="ab" * 32,
        class_id=1,
        label="Shirt",
        confidence=0.89,
        probabilities={"Shirt": 0.89, "T-shirt/top": 0.11}
    )


@pytest.mark.unit
def test_hbase_gateway_publisher_uses_auth_and_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("Basic ")
        assert request.url.path == "/predictions"
        assert b'"schema_version":1' in request.content
        return httpx.Response(201, json={"status": "stored"})

    publisher = HBaseGatewayPublisher(
        "http://hbase-gateway:8080",
        "fashion-api",
        "secret",
        transport=httpx.MockTransport(handler)
    )

    receipt = publisher.publish(event())

    assert receipt.mode == "hbase"
    assert receipt.destination == "http://hbase-gateway:8080"


@pytest.mark.unit
def test_hbase_gateway_publisher_maps_http_failure() -> None:
    publisher = HBaseGatewayPublisher(
        "http://hbase-gateway:8080",
        "fashion-api",
        "secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    )

    with pytest.raises(DeliveryError, match="rejected prediction"):
        publisher.publish(event())
