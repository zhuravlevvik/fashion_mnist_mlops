"""Evaluate the trained classifier and enforce the quality gate."""

from __future__ import annotations

import csv
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from fashion_mnist_mlops.config import CLASS_NAMES, load_config, project_path
from fashion_mnist_mlops.model import load_artifact
from fashion_mnist_mlops.train import load_split

LOGGER = logging.getLogger(__name__)


def evaluate(config_path: str | Path | None = None, enforce_threshold: bool = True) -> dict:
    config = load_config(config_path)
    images, labels = load_split(project_path(config, "processed_dir") / "test.npz")
    artifact = load_artifact(project_path(config, "model_path"))
    predictions = artifact["model"].predict(images)

    minimum = config.getfloat("QUALITY", "min_accuracy")
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "minimum_required_accuracy": minimum,
        "test_samples": int(labels.size),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    metrics_path = project_path(config, "metrics_path")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    matrix_path = project_path(config, "confusion_matrix_path")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("actual", "predicted", "count"))
        writer.writeheader()
        for actual, row in enumerate(matrix):
            for predicted, count in enumerate(row):
                writer.writerow(
                    {
                        "actual": CLASS_NAMES[actual],
                        "predicted": CLASS_NAMES[predicted],
                        "count": int(count),
                    }
                )

    accuracy = metrics["accuracy"]
    f1 = metrics["macro_f1"]
    LOGGER.info(f"Accuracy {accuracy}, macro-F1 {f1}")
    if enforce_threshold and accuracy < minimum:
        raise RuntimeError(f"Quality gate failed: accuracy {accuracy} is below {minimum}")
    return metrics


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.ini")
    parser.add_argument(
        "--allow-below-threshold",
        action="store_true",
        help="Write metrics without failing when accuracy is below the configured gate",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(args.config, enforce_threshold=not args.allow_below_threshold)


if __name__ == "__main__":
    main()
