from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TOPK_COLUMNS = [
    ("pred_top1_index", "pred_top1_prob"),
    ("pred_top2_index", "pred_top2_prob"),
    ("pred_top3_index", "pred_top3_prob"),
    ("pred_top4_index", "pred_top4_prob"),
    ("pred_top5_index", "pred_top5_prob"),
]

SUPPORT_BINS = [
    ("1_to_99", 1, 99),
    ("100_to_999", 100, 999),
    ("1k_to_4_999", 1_000, 4_999),
    ("5k_to_9_999", 5_000, 9_999),
    ("10k_to_49_999", 10_000, 49_999),
    ("50k_to_99_999", 50_000, 99_999),
    ("100k_to_499_999", 100_000, 499_999),
    ("500k_plus", 500_000, None),
]


@dataclass(frozen=True)
class Dirs:
    root: Path
    csv: Path
    richer_slices: Path
    figures: Path
    predictions: Path
    hardware: Path
    hardware_csv: Path
    hardware_figures: Path
    mapping: Path
    mapping_csv: Path
    mapping_figures: Path
    launcher_logs: Path
    logs: Path
    source_refs: Path

def ensure_dirs(output_dir: Path) -> Dirs:
    dirs = Dirs(
        root=output_dir,
        csv=output_dir / "csv",
        richer_slices=output_dir / "csv" / "richer_slices",
        figures=output_dir / "figures",
        predictions=output_dir / "predictions",
        hardware=output_dir / "hardware",
        hardware_csv=output_dir / "hardware" / "csv",
        hardware_figures=output_dir / "hardware" / "figures",
        mapping=output_dir / "mapping",
        mapping_csv=output_dir / "mapping" / "csv",
        mapping_figures=output_dir / "mapping" / "figures",
        launcher_logs=output_dir / "launcher_logs",
        logs=output_dir / "logs",
        source_refs=output_dir / "source_refs",
    )
    for directory in dirs.__dict__.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs

def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)

def write_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def load_class_index(run_dir: Path) -> pd.DataFrame:
    class_index = pd.read_csv(run_dir / "source_refs" / "class_index.csv")
    class_index = class_index.sort_values("class_index").reset_index(drop=True)
    expected = list(range(len(class_index)))
    actual = class_index["class_index"].astype(int).tolist()
    if actual != expected:
        raise ValueError("class_index.csv must be contiguous and zero-based")
    return class_index
