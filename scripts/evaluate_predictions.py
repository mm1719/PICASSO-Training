"""Export source-aware evaluation tables from existing predictions, without a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from scripts.phase4_outputs.common import TOPK_COLUMNS, ensure_dirs
from scripts.phase4_outputs.metrics import (
    export_classification_reports,
    export_confusion_matrices,
    export_table2,
    export_table3,
    export_table4,
    export_threshold_curves,
    topk_arrays,
    topk_hit,
)


def evaluate_predictions(predictions: pd.DataFrame, classes: pd.DataFrame, output: Path) -> dict:
    required = {"image_path", "dataset_id", "valid_status", "invalid_reason", "risk_flag", "true_class_index"}
    required |= {column for pair in TOPK_COLUMNS for column in pair}
    missing = required - set(predictions)
    if missing or predictions.empty:
        raise ValueError(f"Predictions must be nonempty and contain required columns; missing: {sorted(missing)}")
    if not {"class_index", "target_display_name", "train_count"} <= set(classes):
        raise ValueError("Class index requires class_index, target_display_name and train_count.")
    classes = classes.sort_values("class_index").reset_index(drop=True).copy()
    if classes["class_index"].tolist() != list(range(len(classes))) or len(classes) < 5:
        raise ValueError("At least five contiguous zero-based classes are required.")
    if classes["target_display_name"].isna().any() or classes["target_display_name"].duplicated().any():
        raise ValueError("Class display names must be present and unique.")
    support = pd.to_numeric(classes["train_count"], errors="raise")
    if not np.isfinite(support).all() or (support < 1).any() or (support != np.floor(support)).any():
        raise ValueError("Training support must contain positive integers.")
    classes["train_count"] = support
    for column in ["true_class_index", *[column for column, _ in TOPK_COLUMNS]]:
        values = pd.to_numeric(predictions[column], errors="raise")
        if not np.isfinite(values).all() or (values != np.floor(values)).any() or (values < 0).any() or (values >= len(classes)).any():
            raise ValueError("Prediction class indexes must be finite integers within the class index.")
    true, predicted, probability = topk_arrays(predictions)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if (np.diff(probability, axis=1) > 0).any():
        raise ValueError("Top-k probabilities must be in descending order.")
    if any(len(set(row)) != len(row) for row in predicted):
        raise ValueError("Each row must contain five distinct ranked class indexes.")
    output.mkdir(parents=True, exist_ok=False)
    dirs = ensure_dirs(output)
    names = classes["target_display_name"].astype(str).tolist()
    labels = np.arange(len(classes))
    _, _, f1, counts = precision_recall_fscore_support(true, predicted[:, 0], labels=labels, zero_division=0)
    metrics = {
        "sample_count": len(predictions),
        "macro_f1_all_declared_classes": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=counts)),
        "balanced_accuracy_observed_classes": float(balanced_accuracy_score(true, predicted[:, 0])),
        "topk_accuracy": {f"top{k}_accuracy": float(topk_hit(true, predicted, k).mean()) for k in range(1, 6)},
        "support_basis": "post-mapping training split count",
        "metric_scope": "Supplied predictions only; no model or dataset authenticity verification",
    }
    # The historical utility's field name is generic; supply training counts explicitly.
    classes["post_mapping_image_count"] = classes["train_count"]
    export_classification_reports(predictions, names, labels, dirs)
    export_table2(metrics, dirs)
    export_table3(true, predicted, classes, dirs)
    export_table4(true, predicted, names, labels, dirs)
    export_threshold_curves(true, predicted, probability, dirs)
    export_confusion_matrices(true, predicted, names, labels, dirs)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--class-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_predictions(pd.read_csv(args.predictions), pd.read_csv(args.class_index), args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
