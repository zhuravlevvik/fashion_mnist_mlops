"""End-to-end acceptance test for laboratory work 4."""

import os
import time

import httpx
import pytest

from fashion_mnist_mlops.secret_provider import AnsibleVaultSecretProvider


@pytest.mark.integration
def test_kafka_consumer_persists_prediction_in_hbase() -> None:
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    gateway_url = os.getenv("HBASE_GATEWAY_PUBLIC_URL", "http://127.0.0.1:8080")
    credentials = AnsibleVaultSecretProvider(
        os.getenv("ANSIBLE_VAULT_FILE", "secrets/hbase.vault.yml"),
        os.getenv(
            "ANSIBLE_VAULT_PASSWORD_FILE",
            "secrets/ansible-vault-password.txt",
        ),
    ).hbase_gateway_credentials()

    response = httpx.post(
        f"{api_url}/predict",
        json={"pixels": [0] * (28 * 28)},
        timeout=15,
    )
    response.raise_for_status()
    prediction = response.json()

    assert prediction["delivery"] == "kafka"

    stored = None
    for _attempt in range(30):
        stored = httpx.get(
            f"{gateway_url}/predictions/{prediction['prediction_id']}",
            auth=(credentials.username, credentials.password),
            timeout=10,
        )
        if stored.status_code == 200:
            break
        time.sleep(1)

    assert stored is not None
    assert stored.status_code == 200
    assert stored.json()["prediction_id"] == prediction["prediction_id"]
    assert stored.json()["schema_version"] == 1
