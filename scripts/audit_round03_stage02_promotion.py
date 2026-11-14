#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "image_path",
    "dataset_id",
    "source_id",
    "target_id",
    "true_class_index",
    "true_class_name",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paired R3S2 validation predictions.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--high-support-threshold", type=int, default=5000)
    return parser.parse_args()


def load_paired_predictions(baseline_csv: Path, candidate_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = IDENTITY_COLUMNS + ["pred_top1_index"]
    baseline = pd.read_csv(baseline_csv, usecols=usecols)
    candidate = pd.read_csv(candidate_csv, usecols=usecols)
    if baseline.empty or candidate.empty:
        raise ValueError("Prediction exports must be nonempty.")
    if baseline[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError("Prediction identity columns must not contain missing values.")
    if baseline["image_path"].duplicated().any():
        raise ValueError("Prediction image identities must be unique for paired auditing.")
    if len(baseline) != len(candidate):
        raise ValueError(f"Prediction count mismatch: {len(baseline)} != {len(candidate)}")
    if not baseline[IDENTITY_COLUMNS].equals(candidate[IDENTITY_COLUMNS]):
        raise ValueError("Prediction identity/order mismatch between baseline and candidate.")
    if baseline["split"].nunique() != 1 or baseline["split"].iloc[0] != "val":
        raise ValueError("Promotion audit requires validation-only prediction exports.")
    labels = baseline["true_class_index"]
    if sorted(labels.unique().tolist()) != list(range(labels.nunique())):
        raise ValueError("Validation classes must be contiguous and zero-based.")
    if baseline.groupby("true_class_index")["true_class_name"].nunique().max() != 1:
        raise ValueError("Each class index must have one class name.")
    for frame in (baseline, candidate):
        for column in ("true_class_index", "pred_top1_index"):
            values = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(values).all() or (values != np.floor(values)).any():
                raise ValueError("Class indexes must be finite integers.")
            if (values < 0).any() or (values >= labels.nunique()).any():
                raise ValueError("Class index outside the validation target space.")
    return baseline, candidate


def confusion(true: np.ndarray, predicted: np.ndarray, class_count: int) -> np.ndarray:
    return np.bincount(true * class_count + predicted, minlength=class_count * class_count).reshape(
        class_count, class_count
    )


def score_confusion(matrix: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    true_positive = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(
        2 * true_positive,
        2 * true_positive + predicted - true_positive + support - true_positive,
        out=np.zeros_like(true_positive),
        where=(2 * true_positive + predicted - true_positive + support - true_positive) > 0,
    )
    return float(f1.mean()), float(recall.mean()), recall, f1


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.quantile(values, 0.025)), float(np.median(values)), float(np.quantile(values, 0.975))


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def high_support_table(classwise: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "true_class_name",
        "validation_support",
        "recall_delta",
        "recall_ci95_low",
        "recall_ci95_high",
        "f1_delta",
        "f1_ci95_low",
        "f1_ci95_high",
    ]
    return classwise.loc[classwise["high_support"], columns].copy()


def bootstrap_deltas(
    true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    class_count: int,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(true == index) for index in range(class_count)]
    macro_f1 = np.empty(replicates, dtype=float)
    balanced_accuracy = np.empty(replicates, dtype=float)
    recall_delta = np.empty((replicates, class_count), dtype=float)
    f1_delta = np.empty((replicates, class_count), dtype=float)
    for replicate in range(replicates):
        baseline_matrix = np.zeros((class_count, class_count), dtype=np.int64)
        candidate_matrix = np.zeros((class_count, class_count), dtype=np.int64)
        for class_index, indices in enumerate(by_class):
            sampled = indices[rng.integers(0, len(indices), size=len(indices))]
            baseline_matrix[class_index] = np.bincount(
                baseline[sampled], minlength=class_count
            )
            candidate_matrix[class_index] = np.bincount(
                candidate[sampled], minlength=class_count
            )
        base_macro, base_balanced, base_recall, base_f1 = score_confusion(baseline_matrix)
        candidate_macro, candidate_balanced, candidate_recall, candidate_f1 = score_confusion(candidate_matrix)
        macro_f1[replicate] = candidate_macro - base_macro
        balanced_accuracy[replicate] = candidate_balanced - base_balanced
        recall_delta[replicate] = candidate_recall - base_recall
        f1_delta[replicate] = candidate_f1 - base_f1
    return macro_f1, balanced_accuracy, recall_delta, f1_delta


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    baseline_frame, candidate_frame = load_paired_predictions(args.baseline_csv, args.candidate_csv)
    true = baseline_frame["true_class_index"].to_numpy(dtype=int)
    baseline = baseline_frame["pred_top1_index"].to_numpy(dtype=int)
    candidate = candidate_frame["pred_top1_index"].to_numpy(dtype=int)
    class_count = int(true.max()) + 1
    if min(baseline.min(), candidate.min()) < 0 or max(baseline.max(), candidate.max()) >= class_count:
        raise ValueError("Prediction class index falls outside the validation target space.")

    baseline_matrix = confusion(true, baseline, class_count)
    candidate_matrix = confusion(true, candidate, class_count)
    base_macro, base_balanced, base_recall, base_f1 = score_confusion(baseline_matrix)
    candidate_macro, candidate_balanced, candidate_recall, candidate_f1 = score_confusion(candidate_matrix)
    macro_delta, balanced_delta, recall_delta, f1_delta = bootstrap_deltas(
        true,
        baseline,
        candidate,
        class_count,
        args.bootstrap_replicates,
        args.seed,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    classes = (
        baseline_frame[["true_class_index", "true_class_name"]]
        .drop_duplicates()
        .sort_values("true_class_index")
        .reset_index(drop=True)
    )
    support = np.bincount(true, minlength=class_count)
    classwise = classes.assign(
        validation_support=support,
        high_support=support >= args.high_support_threshold,
        baseline_recall=base_recall,
        candidate_recall=candidate_recall,
        recall_delta=candidate_recall - base_recall,
        recall_ci95_low=np.quantile(recall_delta, 0.025, axis=0),
        recall_ci95_high=np.quantile(recall_delta, 0.975, axis=0),
        baseline_f1=base_f1,
        candidate_f1=candidate_f1,
        f1_delta=candidate_f1 - base_f1,
        f1_ci95_low=np.quantile(f1_delta, 0.025, axis=0),
        f1_ci95_high=np.quantile(f1_delta, 0.975, axis=0),
    )
    classwise.to_csv(output_dir / "classwise_delta.csv", index=False)

    global_rows = []
    for metric, baseline_value, candidate_value, estimates in [
        ("macro_f1", base_macro, candidate_macro, macro_delta),
        ("balanced_accuracy", base_balanced, candidate_balanced, balanced_delta),
    ]:
        lower, median, upper = ci95(estimates)
        global_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": candidate_value - baseline_value,
                "bootstrap_ci95_low": lower,
                "bootstrap_median": median,
                "bootstrap_ci95_high": upper,
            }
        )
    global_frame = pd.DataFrame(global_rows)
    global_frame.to_csv(output_dir / "bootstrap_summary.csv", index=False)

    high_support = high_support_table(classwise)
    summary = {
        "baseline_csv": args.baseline_csv.name,
        "candidate_csv": args.candidate_csv.name,
        "sample_count": int(len(baseline_frame)),
        "class_count": class_count,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "high_support_threshold": args.high_support_threshold,
        "global_metrics": global_rows,
        "high_support_classes": high_support.to_dict(orient="records"),
        "alignment_verified": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# Paired Validation Promotion Audit",
        "",
        f"- Paired validation rows: `{len(baseline_frame):,}`",
        f"- Classes: `{class_count}`",
        f"- Class-stratified bootstrap replicates: `{args.bootstrap_replicates}` (seed `{args.seed}`)",
        f"- High-support audit slice: validation support `>= {args.high_support_threshold:,}`",
        "- Prediction identities and order were verified before scoring.",
        "",
        "## Global Deltas",
        "",
        markdown_table(global_frame),
        "",
        "## High-Support Class Deltas",
        "",
        markdown_table(high_support),
        "",
        "## Interpretation Boundary",
        "",
        "Inspect each reported interval and per-class delta. An interval containing zero "
        "does not establish a non-zero paired improvement. This image-row, class-stratified "
        "bootstrap does not measure seed-to-seed variability or account for correlated "
        "images from the same sampling event. It is not a deployment promotion decision.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
