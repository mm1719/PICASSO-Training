from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .common import Dirs, SUPPORT_BINS, TOPK_COLUMNS, read_json


INVALID_REASON_NO_REASON_VALUES = {"", "n/a", "na", "nan", "none", "null"}


def is_no_invalid_reason(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in INVALID_REASON_NO_REASON_VALUES


def topk_arrays(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_lab = predictions["true_class_index"].to_numpy(dtype=int)
    pred_lab = predictions[[col for col, _ in TOPK_COLUMNS]].to_numpy(dtype=int)
    pred_prob = predictions[[col for _, col in TOPK_COLUMNS]].to_numpy(dtype=float)
    return true_lab, pred_lab, pred_prob

def topk_hit(true_lab: np.ndarray, pred_lab: np.ndarray, k: int) -> np.ndarray:
    return (pred_lab[:, :k] == true_lab[:, None]).any(axis=1)

def observed_labels(true_lab: np.ndarray) -> np.ndarray:
    return np.array(sorted(int(label) for label in np.unique(true_lab)), dtype=int)

def export_training_epoch_metrics(run_dir: Path, dirs: Dirs) -> pd.DataFrame:
    epoch_metrics = pd.read_csv(run_dir / "logs" / "epoch_metrics.csv")
    exported = pd.DataFrame(
        {
            "epoch": epoch_metrics["epoch"].astype(int),
            "accuracy": epoch_metrics["train_accuracy"],
            "loss": epoch_metrics["train_loss"],
            "lr": epoch_metrics["lr"],
            "val_accuracy": epoch_metrics["val_accuracy"],
            "val_loss": epoch_metrics["val_loss"],
            "time_seconds": epoch_metrics["seconds"],
        }
    )
    optional_columns = {
        "val_balanced_accuracy": "val_balanced_accuracy",
        "val_macro_f1": "val_macro_f1",
        "val_weighted_f1": "val_weighted_f1",
        "raw_val_accuracy": "raw_val_accuracy",
        "raw_val_loss": "raw_val_loss",
        "raw_val_balanced_accuracy": "raw_val_balanced_accuracy",
        "raw_val_macro_f1": "raw_val_macro_f1",
        "raw_val_weighted_f1": "raw_val_weighted_f1",
    }
    for source_column, output_column in optional_columns.items():
        if source_column in epoch_metrics.columns:
            exported[output_column] = epoch_metrics[source_column]
    exported.to_csv(dirs.csv / "training_epoch_metrics.csv", index=False)
    return exported

def export_classification_reports(
    predictions: pd.DataFrame,
    class_names: list[str],
    labels: np.ndarray,
    dirs: Dirs,
) -> pd.DataFrame:
    true_lab, pred_lab, _ = topk_arrays(predictions)
    report = classification_report(
        true_lab,
        pred_lab[:, 0],
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="label")
    report_df.to_csv(dirs.csv / "classification_report.csv", index=False)

    for column, filename in [
        ("dataset_id", "classification_report_by_dataset.csv"),
        ("valid_status", "classification_report_by_valid_status.csv"),
        ("invalid_reason", "classification_report_by_invalid_reason.csv"),
        ("risk_flag", "classification_report_by_risk_flag.csv"),
    ]:
        rows = []
        for value, subset in predictions.groupby(column, dropna=False, sort=True):
            if column == "invalid_reason" and is_no_invalid_reason(value):
                continue
            sub_true, sub_pred, _ = topk_arrays(subset)
            sub_labels = observed_labels(sub_true)
            sub_class_names = [class_names[label] for label in sub_labels]
            sub_report = classification_report(
                sub_true,
                sub_pred[:, 0],
                labels=sub_labels,
                target_names=sub_class_names,
                zero_division=0,
                output_dict=True,
            )
            sub_df = pd.DataFrame(sub_report).transpose().reset_index(names="label")
            sub_df.insert(0, "slice_value", value)
            sub_df.insert(0, "slice_name", column)
            sub_df.insert(2, "slice_sample_count", int(len(subset)))
            sub_df.insert(3, "slice_observed_label_count", int(len(sub_labels)))
            rows.append(sub_df)
        if rows:
            output = pd.concat(rows, ignore_index=True)
        else:
            output = pd.DataFrame(
                columns=[
                    "slice_name",
                    "slice_value",
                    "slice_sample_count",
                    "slice_observed_label_count",
                    "label",
                    "precision",
                    "recall",
                    "f1-score",
                    "support",
                ]
            )
        output.to_csv(dirs.richer_slices / filename, index=False)
    return report_df

def export_table2(metrics: dict, dirs: Dirs) -> pd.DataFrame:
    topk_metrics = metrics.get("topk_accuracy", metrics)
    row = {"metric": "accuracy_pct"}
    for k in range(1, 6):
        row[f"top_{k}"] = topk_metrics[f"top{k}_accuracy"] * 100.0
    table2 = pd.DataFrame([row])
    table2.to_csv(dirs.csv / "table2_topk_metrics.csv", index=False)
    return table2

def export_table3(
    true_lab: np.ndarray,
    pred_lab: np.ndarray,
    class_index: pd.DataFrame,
    dirs: Dirs,
) -> pd.DataFrame:
    rows = []
    for group_name, lower, upper in SUPPORT_BINS:
        if upper is None:
            class_subset = class_index[class_index["post_mapping_image_count"] >= lower]
        else:
            class_subset = class_index[
                (class_index["post_mapping_image_count"] >= lower)
                & (class_index["post_mapping_image_count"] <= upper)
            ]
        if class_subset.empty:
            continue
        class_ids = class_subset["class_index"].astype(int).to_numpy()
        mask = np.isin(true_lab, class_ids)
        if not mask.any():
            continue
        rows.append(
            {
                "sample_range": group_name,
                "num_classes": int(len(class_subset)),
                "num_samples": int(mask.sum()),
                "top1_accuracy_pct": 100.0 * float(topk_hit(true_lab[mask], pred_lab[mask], 1).mean()),
                "top3_accuracy_pct": 100.0 * float(topk_hit(true_lab[mask], pred_lab[mask], 3).mean()),
                "top5_accuracy_pct": 100.0 * float(topk_hit(true_lab[mask], pred_lab[mask], 5).mean()),
            }
        )
    table3 = pd.DataFrame(rows)
    table3.to_csv(dirs.csv / "table3_grouped_accuracy.csv", index=False)
    return table3

def export_table4(
    true_lab: np.ndarray,
    pred_lab: np.ndarray,
    class_names: list[str],
    labels: np.ndarray,
    dirs: Dirs,
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_lab,
        pred_lab[:, 0],
        labels=labels,
        zero_division=0,
    )
    table4 = pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )
    table4["precision_pct"] = table4["precision"] * 100.0
    table4["recall_pct"] = table4["recall"] * 100.0
    table4["f1_score_pct"] = table4["f1_score"] * 100.0
    table4 = table4.sort_values("f1_score_pct", ascending=False).reset_index(drop=True)
    table4.to_csv(dirs.csv / "table4_classwise_performance.csv", index=False)
    return table4

def export_threshold_curves(
    true_lab: np.ndarray,
    pred_lab: np.ndarray,
    pred_prob: np.ndarray,
    dirs: Dirs,
) -> pd.DataFrame:
    rows = []
    thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)
    top1_prob = pred_prob[:, 0]
    for k in range(1, 6):
        for threshold in thresholds:
            retained_mask = top1_prob >= threshold
            retained_count = int(retained_mask.sum())
            if retained_count == 0:
                correct_count = 0
                proportion_correct = math.nan
            else:
                correct = topk_hit(true_lab[retained_mask], pred_lab[retained_mask], k)
                correct_count = int(correct.sum())
                proportion_correct = float(correct_count / retained_count)
            rows.append(
                {
                    "threshold": threshold,
                    "top_k": k,
                    "retained_count": retained_count,
                    "correct_count": correct_count,
                    "proportion_correct": proportion_correct,
                }
            )
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(dirs.csv / "figure4_threshold_curves.csv", index=False)
    return threshold_df

def export_confusion_matrices(
    true_lab: np.ndarray,
    pred_lab: np.ndarray,
    class_names: list[str],
    labels: np.ndarray,
    dirs: Dirs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cm = confusion_matrix(true_lab, pred_lab[:, 0], labels=labels, normalize="true")
    cm_zero = cm.copy()
    np.fill_diagonal(cm_zero, 0.0)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_zero_df = pd.DataFrame(cm_zero, index=class_names, columns=class_names)
    cm_df.to_csv(dirs.csv / "figure5_confusion_matrix_normalized.csv")
    cm_zero_df.to_csv(dirs.csv / "figure6_confusion_matrix_zero_diagonal.csv")
    return cm_df, cm_zero_df

def export_richer_slice_support(run_dir: Path, dirs: Dirs) -> None:
    source = run_dir / "predictions" / "group_accuracy_by_support.csv"
    if source.exists():
        shutil.copy2(source, dirs.richer_slices / "group_accuracy_by_support_detailed.csv")

    target_map = pd.read_csv(run_dir / "source_refs" / "target_id_map.csv", dtype=str)
    target_map["source_image_count"] = pd.to_numeric(
        target_map["source_image_count"],
        errors="coerce",
    ).fillna(0)
    outcomes = []
    for row in target_map.itertuples(index=False):
        status = str(row.status).strip().lower()
        if status == "dropped":
            outcome = "dropped"
        elif str(row.source_id).strip() == str(row.target_id).strip():
            outcome = "accepted"
        else:
            outcome = "merged"
        outcomes.append(outcome)
    target_map["mapping_outcome"] = outcomes
    rows = []
    for outcome in ["accepted", "merged", "dropped"]:
        subset = target_map[target_map["mapping_outcome"] == outcome]
        rows.append(
            {
                "mapping_outcome": outcome,
                "source_target_count": int(len(subset)),
                "image_count": int(subset["source_image_count"].sum()),
            }
        )
    pd.DataFrame(rows).to_csv(
        dirs.richer_slices / "mapping_outcome_summary.csv",
        index=False,
    )

    stale = dirs.richer_slices / "mapping_status_summary.csv"
    if stale.exists():
        stale.unlink()
