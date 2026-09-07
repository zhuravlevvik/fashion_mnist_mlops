"""Create local Ansible Vault material without secrets."""

from __future__ import annotations

import argparse
import os
import secrets
import stat
import subprocess
import tempfile
from pathlib import Path

import yaml

from fashion_mnist_mlops.secret_provider import (
    ansible_subprocess_environment,
    ansible_vault_executable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault-file", default="secrets/hbase.vault.yml")
    parser.add_argument("--password-file", default="secrets/ansible-vault-password.txt")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_password_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"{secrets.token_urlsafe(48)}\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main() -> None:
    args = parse_args()
    vault_path = Path(args.vault_file)
    password_path = Path(args.password_file)
    if vault_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing vault: {vault_path}")

    ensure_password_file(password_path)
    payload = {
        "hbase_gateway": {
            "username": os.getenv("HBASE_GATEWAY_USERNAME", "fashion-api"),
            "password": os.getenv("HBASE_GATEWAY_PASSWORD", secrets.token_urlsafe(32)),
        }
    }

    vault_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as source:
        yaml.safe_dump(payload, source, sort_keys=True)
        source_path = Path(source.name)
    try:
        subprocess.run(
            [
                ansible_vault_executable(),
                "encrypt",
                "--vault-password-file",
                str(password_path),
                "--output",
                str(vault_path),
                str(source_path),
            ],
            check=True,
            env=ansible_subprocess_environment(),
        )
    finally:
        source_path.unlink(missing_ok=True)

    print(f"Created encrypted vault: {vault_path}")
    print(f"Created ignored password file: {password_path}")


if __name__ == "__main__":
    main()
