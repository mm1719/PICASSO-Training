"""Exercise data contracts and evaluation with invented rows, not images or a model."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import audit_round03_stage02_promotion as audit
from scripts import prepare_round00_stage01_data as prepare
from scripts.evaluate_predictions import evaluate_predictions


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_frame(split_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, source in enumerate(split_manifest.to_dict(orient="records")):
        true = int(source["class_index"])
        top = true if i % 3 else (true + 1) % 5
        ranking = [top, *[index for index in range(5) if index != top]]
        row = {**source, "true_class_index": true, "true_class_name": f"synthetic_class_{true}"}
        for k, (index, probability) in enumerate(zip(ranking, [0.60, 0.20, 0.10, 0.06, 0.04]), 1):
            row[f"pred_top{k}_index"] = index
            row[f"pred_top{k}_prob"] = probability
        rows.append(row)
    return pd.DataFrame(rows)


def run_demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "synthetic_inputs"
    inputs.mkdir()
    rows = []
    profiles = []
    for index in range(5):
        target = str(100 + index)
        profiles.append({"aphia_id": target, "class_display_name": f"synthetic_class_{index}", "valid_statuses": "valid", "image_count": 20})
        for sample in range(20):
            rows.append({
                "image_path": f"synthetic/class_{index}/item_{sample}.png",
                "dataset_id": f"synthetic_source_{sample % 2}",
                "valid_status": "valid" if sample % 2 else "invalid",
                "invalid_reason": "n/a" if sample % 2 else "synthetic_unknown",
                "risk_flag": "n/a" if sample % 3 else "synthetic_risk",
                "aphia_id": target,
                "class_display_name": f"synthetic_class_{index}",
                "label_original": f"invented_label_{index}", "width": 16, "height": 16,
            })
    manifest = inputs / "manifest.csv.gz"
    with gzip.open(manifest, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_csv(inputs / "profiles.csv", profiles)
    write_csv(inputs / "mapping.csv", [], ["mapping_status", "source_aphia_id", "target_aphia_id"])
    support = output / "supporting"
    args = ["prepare", "--manifest", str(manifest), "--target-profiles", str(inputs / "profiles.csv"), "--before-training-mapping", str(inputs / "mapping.csv"), "--output-dir", str(support), "--expected-manifest-sha256", prepare.sha256_file(manifest)]
    with patch.object(sys, "argv", args):
        prepare.main()
    split = pd.read_csv(support / "split_manifest.csv")
    predictions = prediction_frame(split[split["split"] == "val"])
    metrics = evaluate_predictions(predictions, pd.read_csv(support / "class_index.csv"), output / "evaluation")
    predictions.to_csv(output / "synthetic_baseline.csv", index=False)
    # Identical models give exactly zero paired deltas; this is a correctness fixture.
    predictions.to_csv(output / "synthetic_candidate.csv", index=False)
    with patch.object(sys, "argv", ["audit", "--baseline-csv", str(output / "synthetic_baseline.csv"), "--candidate-csv", str(output / "synthetic_candidate.csv"), "--output-dir", str(output / "paired_audit"), "--bootstrap-replicates", "25"]):
        audit.main()
    report = {"synthetic_only": True, "model_created": False, "images_created_or_opened": False, "manifest_rows": len(rows), "split_counts": split["split"].value_counts().to_dict(), "evaluation_rows": metrics["sample_count"]}
    (output / "SYNTHETIC_ONLY.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
