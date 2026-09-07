"""End-to-end acceptance test for laboratory work 3."""

import os

import httpx
import pytest

from fashion_mnist_mlops.secret_provider import AnsibleVaultSecretProvider


@pytest.mark.integration
def test_vault_credentials_protect_hbase_round_trip() -> None:
    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    gateway_url = os.getenv("HBASE_GATEWAY_PUBLIC_URL", "http://127.0.0.1:8080")

    provider = AnsibleVaultSecretProvider(
        os.getenv("ANSIBLE_VAULT_FILE", "secrets/hbase.vault.yml"),
        os.getenv(
            "ANSIBLE_VAULT_PASSWORD_FILE",
            "secrets/ansible-vault-password.txt",
        ),
    )
    credentials = provider.hbase_gateway_credentials()

    response = httpx.post(f"{api_url}/predict", json={"pixels": [0] * (28 * 28)}, timeout=15)
    response.raise_for_status()
    prediction = response.json()

    unauthorized = httpx.get(
        f"{gateway_url}/predictions/{prediction['prediction_id']}",
        timeout=15,
    )
    authorized = httpx.get(
        f"{gateway_url}/predictions/{prediction['prediction_id']}",
        auth=(credentials.username, credentials.password),
        timeout=15,
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["prediction_id"] == prediction["prediction_id"]
