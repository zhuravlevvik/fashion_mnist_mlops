"""End-to-end acceptance test for lab2."""

import os

import httpx
import pytest


@pytest.mark.integration
def test_prediction_is_persisted_in_hbase() -> None:
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    gateway_url = os.getenv("HBASE_GATEWAY_PUBLIC_URL", "http://127.0.0.1:8080")
    auth = (
        os.environ["HBASE_GATEWAY_USERNAME"],
        os.environ["HBASE_GATEWAY_PASSWORD"],
    )
    payload = {"pixels": [0] * (28 * 28)}

    response = httpx.post(f"{api_url}/predict", json=payload, timeout=15)
    response.raise_for_status()
    result = response.json()

    stored = httpx.get(
        f"{gateway_url}/predictions/{result['prediction_id']}",
        auth=auth,
        timeout=15,
    )
    stored.raise_for_status()

    assert result["delivery"] == "hbase"
    assert stored.json()["prediction_id"] == result["prediction_id"]
    assert stored.json()["label"] == result["label"]
