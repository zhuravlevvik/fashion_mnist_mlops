from types import SimpleNamespace

import pytest

from fashion_mnist_mlops.secret_provider import (
    AnsibleVaultSecretProvider,
    EnvironmentSecretProvider,
    GatewayCredentials,
    SecretProviderError,
    get_secret_provider,
)


@pytest.mark.unit
def test_environment_secret_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HBASE_GATEWAY_USERNAME", "fashion-api")
    monkeypatch.setenv("HBASE_GATEWAY_PASSWORD", "secret")

    assert EnvironmentSecretProvider().hbase_gateway_credentials() == GatewayCredentials(
        username="fashion-api", password="secret"
    )


@pytest.mark.unit
def test_ansible_vault_provider_plaintext_in_memory() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs):
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return SimpleNamespace(
            stdout="hbase_gateway:\n  username: fashion-api\n  password: secret\n"
        )

    provider = AnsibleVaultSecretProvider("vault.yml", "password.txt", runner=runner)

    assert provider.hbase_gateway_credentials().password == "secret"
    assert calls[0][0].endswith("ansible-vault")
    assert calls[0][1] == "view"


@pytest.mark.unit
def test_ansible_vault_provider_rejects_invalid_document() -> None:
    provider = AnsibleVaultSecretProvider(
        "vault.yml",
        "password.txt",
        runner=lambda *_args, **kwargs: SimpleNamespace(stdout="other: value\n"),
    )

    with pytest.raises(SecretProviderError, match="Unable to load"):
        provider.hbase_gateway_credentials()


@pytest.mark.unit
def test_unknown_secret_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_PROVIDER", "unknown")
    get_secret_provider.cache_clear()

    with pytest.raises(SecretProviderError, match="Unsupported"):
        get_secret_provider()
