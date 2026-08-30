from pathlib import Path

import pytest

from fashion_mnist_mlops.config import load_config, project_path


@pytest.mark.unit
def test_default_config_has_resolvable_path() -> None:
    config = load_config()
    assert project_path(config, "model_path").is_absolute()
    assert project_path(config, "model_path").name == "model.joblib"


@pytest.mark.unit
def test_missing_config_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.ini")
