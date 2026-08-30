"""Download and prepare the Fashion-MNIST IDX files."""

from __future__ import annotations

import gzip
import logging
import os
import struct
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from fashion_mnist_mlops.config import load_config, project_path

LOGGER = logging.getLogger(__name__)

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, destination: Path) -> None:
    """Download a file atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info(f"Using existing file {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".part")
    LOGGER.info(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def download_dataset(config_path: str | Path | None = None) -> Path:
    """Download the four canonical Fashion-MNIST archives."""

    config = load_config(config_path)
    raw_dir = project_path(config, "raw_dir")
    base_url = config.get("DATA", "base_url").rstrip("/")
    for filename in FILES.values():
        download_file(f"{base_url}/{filename}", raw_dir / filename)
    return raw_dir


def read_idx_images(path: Path) -> np.ndarray:
    """Read an IDX3 image archive and return uint8 images."""

    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected IDX image magic {magic} in {path}")
        payload = stream.read()

    expected = count * rows * columns
    if len(payload) != expected:
        raise ValueError(f"Expected {expected} image bytes in {path}, got {len(payload)}")

    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns)


def read_idx_labels(path: Path) -> np.ndarray:
    """Read an IDX1 label archive and return uint8 labels."""

    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label magic {magic} in {path}")
        payload = stream.read()

    if len(payload) != count:
        raise ValueError(f"Expected {count} labels in {path}, got {len(payload)}")
    return np.frombuffer(payload, dtype=np.uint8)


def stratified_subset(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Selects a deterministic, approximately class-balanced subset."""

    if limit <= 0 or limit >= labels.size:
        return np.arange(labels.size)

    classes = np.unique(labels)
    per_class, remainder = divmod(limit, classes.size)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []

    for position, label in enumerate(classes):
        class_indices = np.flatnonzero(labels == label)
        take = per_class + (1 if position < remainder else 0)
        if take > class_indices.size:
            raise ValueError(f"Class {label} contains only {class_indices.size} samples")
        selected.append(rng.choice(class_indices, size=take, replace=False))

    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def prepare_dataset(config_path: str | Path | None = None) -> Path:
    """Parse, validate, subset and persist train/test arrays as compressed NPZ."""

    config = load_config(config_path)
    raw_dir = project_path(config, "raw_dir")
    processed_dir = project_path(config, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_images = read_idx_images(raw_dir / FILES["train_images"])
    train_labels = read_idx_labels(raw_dir / FILES["train_labels"])
    test_images = read_idx_images(raw_dir / FILES["test_images"])
    test_labels = read_idx_labels(raw_dir / FILES["test_labels"])

    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("Train image and label counts differ")
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("Test image and label counts differ")
    if train_images.shape[1:] != (28, 28) or test_images.shape[1:] != (28, 28):
        raise ValueError("Fashion-MNIST images must be 28x28 pixels")

    seed = config.getint("DATA", "random_seed")
    train_indices = stratified_subset(train_labels, config.getint("DATA", "train_limit"), seed)
    test_indices = stratified_subset(test_labels, config.getint("DATA", "test_limit"), seed)

    np.savez_compressed(
        processed_dir / "train.npz",
        images=train_images[train_indices],
        labels=train_labels[train_indices],
    )
    np.savez_compressed(
        processed_dir / "test.npz",
        images=test_images[test_indices],
        labels=test_labels[test_indices],
    )
    LOGGER.info(f"Prepared {train_indices.size} train and {test_indices.size} test samples")

    return processed_dir


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "all"))
    parser.add_argument("--config", default=None, help="Path to config.ini")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.command in {"download", "all"}:
        download_dataset(args.config)
    if args.command in {"prepare", "all"}:
        prepare_dataset(args.config)


if __name__ == "__main__":
    main()
