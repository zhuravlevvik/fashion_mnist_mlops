"""Configuration helpers shared by training, evaluation and inference."""

from __future__ import annotations

import os
from configparser import ConfigParser
from pathlib import Path

SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(SOURCE_PROJECT_ROOT))).expanduser().resolve()

DEFAULT_CONFIG_PATH = (
    Path(os.environ.get("CONFIG_PATH", str(PROJECT_ROOT / "config.ini"))).expanduser().resolve()
)

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_config(path: str | Path | None = None) -> ConfigParser:
    """Load and validate the profect INI configuration."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()

    parser = ConfigParser()
    loaded = parser.read(config_path)
    if not loaded:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    requires_sections = {"PATHS", "DATA", "MODEL", "QUALITY", "API"}
    missing = requires_sections.difference(parser.sections())
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    parser["DEFAULT"]["project_root"] = str(PROJECT_ROOT)
    parser["DEFAULT"]["config_path"] = str(config_path)
    return parser


def project_path(config: ConfigParser, option: str) -> Path:
    """Resolve a configured path relative to the repository root."""

    value = Path(config.get("PATHS", option)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value
