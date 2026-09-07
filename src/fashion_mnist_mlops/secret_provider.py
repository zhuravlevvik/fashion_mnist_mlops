"""Secret-provider abstraction with Ansible Vault support."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml


class SecretProviderError(RuntimeError):
    """Raised when required secrets cannot be resolved safely."""


@dataclass(frozen=True)
class GatewayCredentials:
    username: str
    password: str


class SecretProvider(Protocol):
    def hbase_gateway_credentials(self) -> GatewayCredentials:
        ...


def ansible_vault_executable() -> str:
    adjacent = Path(sys.executable).with_name("ansible-vault")
    if adjacent.is_file():
        return str(adjacent)
    executable = shutil.which("ansible-vault")
    if executable:
        return executable
    raise SecretProviderError("ansible-vault executable is unavailable")


def ansible_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("ANSIBLE_HOME", "/tmp/ansible-home")
    environment.setdefault("ANSIBLE_LOCAL_TEMP", "/tmp/ansible-local")
    return environment


def validate_credentials(credentials: GatewayCredentials) -> GatewayCredentials:
    if not credentials.username.strip() or not credentials.password:
        raise SecretProviderError("HBase gateway credentials must be non-empty")
    return credentials


class EnvironmentSecretProvider:
    def hbase_gateway_credentials(self) -> GatewayCredentials:
        try:
            credentials = GatewayCredentials(
                username=os.environ["HBASE_GATEWAY_USERNAME"],
                password=os.environ["HBASE_GATEWAY_PASSWORD"],
            )
        except KeyError as e:
            raise SecretProviderError(f"Missing environment secret: {e.args[0]}") from e
        return validate_credentials(credentials)


class AnsibleVaultSecretProvider:
    """Decrypts a vault YAML document to subprocess memory only."""

    def __init__(
        self,
        vault_file: str | Path,
        password_file: str | Path,
        runner=subprocess.run,
    ) -> None:
        self.vault_file = Path(vault_file)
        self.password_file = Path(password_file)
        self._runner = runner

    def hbase_gateway_credentials(self) -> GatewayCredentials:
        command = [
            ansible_vault_executable(),
            "view",
            "--vault-password-file",
            str(self.password_file),
            str(self.vault_file),
        ]

        try:
            result = self._runner(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
                env=ansible_subprocess_environment(),
            )
            payload = yaml.safe_load(result.stdout)
            hbase = payload["hbase_gateway"]
            credentials = GatewayCredentials(
                username=hbase["username"],
                password=hbase["password"],
            )
        except Exception as e:
            raise SecretProviderError(
                "Unable to load HBase credentials from Ansible Vault"
            ) from e
        return validate_credentials(credentials)


@lru_cache(maxsize=1)
def get_secret_provider() -> SecretProvider:
    provider = os.getenv("SECRET_PROVIDER", "environment").strip().lower()
    if provider == "environment":
        return EnvironmentSecretProvider()
    if provider == "ansible-vault":
        return AnsibleVaultSecretProvider(
            vault_file=os.getenv("ANSIBLE_VAULT_FILE", "/run/config/hbase.vault.yml"),
            password_file=os.getenv(
                "ANSIBLE_VAULT_PASSWORD_FILE",
                "/run/secrets/ansible_vault_password",
            ),
        )
    raise SecretProviderError(f"Unsupported SECRET_PROVIDER: {provider}")


@lru_cache(maxsize=1)
def get_hbase_gateway_credentials() -> GatewayCredentials:
    return get_secret_provider().hbase_gateway_credentials()
