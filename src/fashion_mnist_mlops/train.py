"""Train a Fashion-MNIST classifier."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from fashion_mnist_mlops.config import CLASS_NAMES, load_config, project_path
from fashion_mnist_mlops.model import build_model

LOGGER = logging.getLogger(__name__)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return payload["images"], payload["labels"]


def train(config_path: str | Path | None = None) -> Path:
    config = load_config(config_path)
    config_file = Path(config["DEFAULT"]["config_path"])
    processed_dir = project_path(config, "processed_dir")
    model_path = project_path(config, "model_path")
    summary_path = project_path(config, "train_summary_path")

    images, labels = load_split(processed_dir / "train.npz")
    model = build_model(config)

    started = time.perf_counter()
    model.fit(images, labels)
    duration = time.perf_counter() - started

    metadata = {
        "algorithm": "SGDClassifier",
        "loss": config["MODEL"].get("loss"),
        "classes": list(CLASS_NAMES),
        "config_sha256": hashlib.sha3_256(config_file.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_shape": [28, 28],
        "scikit_learn_version": sklearn.__version__,
        "train_samples": int(labels.size),
    }

    artifact = {"model": model, "metadata": metadata}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)

    summary = {**metadata, "duration_seconds": round(duration, 3)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    LOGGER.info(f"Model saved to {model_path} after {duration} seconds")
    return model_path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.ini")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
