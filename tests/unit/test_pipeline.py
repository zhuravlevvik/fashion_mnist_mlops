import gzip
import struct
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pytest

from fashion_mnist_mlops.data import FILES, download_dataset, prepare_dataset
from fashion_mnist_mlops.evaluate import evaluate
from fashion_mnist_mlops.model import load_artifact
from fashion_mnist_mlops.train import train


def synthetic_fashion(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10, dtype=np.uint8), samples_per_class)
    images = np.zeros((labels.size, 28, 28), dtype=np.float32)
    for index, label in enumerate(labels):
        start = int(label) * 2 + 2
        images[index, :, start : start + 2] = 255
        images[index, start : start + 2, :] = 180
    images += rng.normal(0, 2, size=images.shape)
    return np.clip(images, 0, 255).astype(np.uint8), labels


def write_idx_dataset(directory: Path) -> None:
    directory.mkdir(parents=True)
    train_images, train_labels = synthetic_fashion(12, seed=1)
    test_images, test_labels = synthetic_fashion(4, seed=2)

    payloads = {
        FILES["train_images"]: (">IIII", (2051, *train_images.shape), train_images),
        FILES["train_labels"]: (">II", (2049, train_labels.size), train_labels),
        FILES["test_images"]: (">IIII", (2051, *test_images.shape), test_images),
        FILES["test_labels"]: (">II", (2049, test_labels.size), test_labels),
    }

    for filename, (header_format, header, values) in payloads.items():
        with gzip.open(directory / filename, "wb") as stream:
            stream.write(struct.pack(header_format, *header))
            stream.write(values.tobytes())


def write_config(path: Path, source: Path, workspace: Path) -> None:
    config = ConfigParser()
    config.read_dict(
        {
            "PATHS": {
                "raw_dir": str(workspace / "raw"),
                "processed_dir": str(workspace / "processed"),
                "model_path": str(workspace / "models/model.joblib"),
                "metrics_path": str(workspace / "reports/metrics.json"),
                "confusion_matrix_path": str(workspace / "reports/confusion_matrix.csv"),
                "train_summary_path": str(workspace / "reports/train_summary.json"),
            },
            "DATA": {
                "base_url": source.as_uri(),
                "train_limit": "100",
                "test_limit": "30",
                "random_seed": "19",
            },
            "MODEL": {
                "loss": "log_loss",
                "penalty": "l2",
                "alpha": "0.0001",
                "max_iter": "200",
                "early_stopping": "false",
                "validation_fraction": "0.1",
                "n_iter_no_change": "5",
                "random_seed": "19",
            },
            "QUALITY": {"min_accuracy": "0.0"},
            "API": {"host": "127.0.0.1", "port": "8000"},
        }
    )
    with path.open("w") as output:
        config.write(output)


@pytest.mark.unit
def test_complete_training_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.ini"
    write_idx_dataset(source)
    write_config(config_path, source, workspace)

    raw_dir = download_dataset(config_path)
    assert len(list(raw_dir.glob("*.gz"))) == 4
    download_dataset(config_path)
    processed_dir = prepare_dataset(config_path)
    assert (processed_dir / "train.npz").exists()

    model_path = train(config_path)
    artifact = load_artifact(model_path)
    assert artifact["metadata"]["train_samples"] == 100
    assert artifact["model"].named_steps["classifier"].random_state == 19

    metrics = evaluate(config_path)
    assert metrics["accuracy"] >= 0.5
    assert (workspace / "reports/confusion_matrix.csv").exists()
