import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from fashion_mnist_mlops.data import read_idx_images, read_idx_labels, stratified_subset


def write_idx_images(path: Path, images: np.ndarray, magic: int = 2051) -> None:
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">IIII", magic, *images.shape))
        stream.write(images.astype(np.uint8).tobytes())


def write_idx_labels(path: Path, labels: np.ndarray, magic: int = 2049) -> None:
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">II", magic, labels.size))
        stream.write(labels.astype(np.uint8).tobytes())


@pytest.mark.unit
def test_idx_roundtrip(tmp_path: Path) -> None:
    images = np.arange(3 * 28 * 28, dtype=np.uint8).reshape(3, 28, 28)
    labels = np.array([1, 4, 9], dtype=np.uint8)
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    write_idx_images(image_path, images)
    write_idx_labels(label_path, labels)

    np.testing.assert_array_equal(read_idx_images(image_path), images)
    np.testing.assert_array_equal(read_idx_labels(label_path), labels)


@pytest.mark.unit
def test_idx_rejects_wrong_magic(tmp_path: Path) -> None:
    path = tmp_path / "images.gz"
    write_idx_images(path, np.zeros((1, 28, 28), dtype=np.uint8), magic=19)
    with pytest.raises(ValueError, match="magic"):
        read_idx_images(path)


@pytest.mark.unit
def test_stratified_subset_is_balanced_and_deterministic() -> None:
    labels = np.repeat(np.arange(10), 20)
    first = stratified_subset(labels, limit=50, seed=19)
    second = stratified_subset(labels, limit=50, seed=19)

    np.testing.assert_array_equal(first, second)
    assert np.bincount(labels[first], minlength=10).tolist() == [5] * 10
