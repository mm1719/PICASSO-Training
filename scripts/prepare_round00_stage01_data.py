#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260421
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
TAIPEI_PHASE = "phase4"
ROUND = "round00"
STAGE = "stage01"
LINE_CODE = "best_anchor_integrated_demo"
OLD_DROPPED_STATUS = "unresolved_target_not_in_ml_ready"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_target_profiles(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["aphia_id"]: row for row in csv.DictReader(handle)}


def read_mapping(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], Counter]:
    accepted: dict[str, dict[str, str]] = {}
    dropped: dict[str, dict[str, str]] = {}
    outcome_counts: Counter = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            status = row["mapping_status"]
            source_id = row["source_aphia_id"]
            if status == "accepted":
                accepted[source_id] = row
                outcome_counts["merged"] += 1
            else:
                dropped[source_id] = row
                outcome_counts["dropped"] += 1
                row["notes"] = (
                    f"Original mapping status: {status}. {row.get('notes', '')}"
                ).strip()
    return accepted, dropped, outcome_counts


def mapped_target_id(source_id: str, accepted_mapping: dict[str, dict[str, str]]) -> str:
    row = accepted_mapping.get(source_id)
    if row is None:
        return source_id
    return row["target_aphia_id"]


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int, str]:
    train = int(total * ratios[0])
    val = int(total * ratios[1])
    test = total - train - val
    fallback = "phase2_floor_allocation"

    # Keep phase 2 floor allocation for ordinary classes, but do not create a
    # trainable output class with zero train examples.
    if total > 0 and train == 0:
        train = 1
        if test > 0:
            test -= 1
        elif val > 0:
            val -= 1
        fallback = "ensure_min_one_train_example"
    return train, val, test, fallback


def split_row_ids(
    grouped_ids: dict[str, list[int]],
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[dict[int, str], dict[str, dict[str, Any]], Counter]:
    rng = random.Random(seed)
    assignments: dict[int, str] = {}
    per_target: dict[str, dict[str, Any]] = {}
    fallback_counts: Counter = Counter()

    for target_id in sorted(grouped_ids):
        row_ids = grouped_ids[target_id][:]
        rng.shuffle(row_ids)
        train_count, val_count, test_count, fallback = allocate_counts(len(row_ids), ratios)
        fallback_counts[fallback] += 1

        train_ids = row_ids[:train_count]
        val_ids = row_ids[train_count : train_count + val_count]
        test_ids = row_ids[train_count + val_count :]
        for row_id in train_ids:
            assignments[row_id] = "train"
        for row_id in val_ids:
            assignments[row_id] = "val"
        for row_id in test_ids:
            assignments[row_id] = "test"

        per_target[target_id] = {
            "total": len(row_ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "allocation_rule": fallback,
        }

    return assignments, per_target, fallback_counts


def classify_target_kind(target_id: str, source_ids: set[str], profiles: dict[str, dict[str, str]]) -> str:
    if len(source_ids) > 1:
        return "merged"
    profile = profiles.get(target_id)
    if profile and profile.get("valid_statuses") == "valid":
        return "aphia"
    if target_id.isdigit():
        return "aphia"
    return "pseudo_aphia"


def write_target_mapping(
    path: Path,
    profiles: dict[str, dict[str, str]],
    accepted_mapping: dict[str, dict[str, str]],
    dropped_mapping: dict[str, dict[str, str]],
) -> None:
    fieldnames = [
        "source_id",
        "target_id",
        "source_display_name",
        "target_display_name",
        "mapping_scope",
        "effective_phase",
        "effective_round",
        "effective_stage",
        "line_code",
        "mapping_reason",
        "status",
        "source_image_count",
        "target_image_count_before_round",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_id in sorted(profiles):
            profile = profiles[source_id]
            accepted = accepted_mapping.get(source_id)
            dropped = dropped_mapping.get(source_id)
            if accepted is not None:
                target_id = accepted["target_aphia_id"]
                target_display = accepted["target_class_display_name"]
                status = "active"
                reason = accepted["mapping_reason"]
                notes = accepted.get("notes", "")
                scope = accepted["mapping_scope"]
                target_count = accepted["target_image_count"]
            elif dropped is not None:
                target_id = dropped["target_aphia_id"] or ""
                target_display = dropped["target_class_display_name"] or ""
                status = "dropped"
                reason = dropped["mapping_reason"]
                notes = dropped.get("notes", "")
                scope = dropped["mapping_scope"]
                target_count = dropped["target_image_count"]
            else:
                target_id = source_id
                target_display = profile["class_display_name"]
                status = "active"
                reason = "identity_mapping"
                notes = ""
                scope = "identity"
                target_count = profile["image_count"]

            writer.writerow(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_display_name": profile["class_display_name"],
                    "target_display_name": target_display,
                    "mapping_scope": scope,
                    "effective_phase": TAIPEI_PHASE,
                    "effective_round": ROUND,
                    "effective_stage": STAGE,
                    "line_code": LINE_CODE,
                    "mapping_reason": reason,
                    "status": status,
                    "source_image_count": profile["image_count"],
                    "target_image_count_before_round": target_count,
                    "notes": notes,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare round 0 stage 1 supporting data.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-profiles", type=Path, required=True)
    parser.add_argument(
        "--before-training-mapping",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(not 0 <= ratio <= 1 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    manifest_hash = sha256_file(args.manifest)
    if args.expected_manifest_sha256 and manifest_hash != args.expected_manifest_sha256:
        raise ValueError(
            f"Manifest checksum mismatch: expected {args.expected_manifest_sha256}, got {manifest_hash}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    profiles = read_target_profiles(args.target_profiles)
    accepted_mapping, dropped_mapping, mapping_outcome_counts = read_mapping(args.before_training_mapping)
    write_target_mapping(args.output_dir / "target_id_map.csv", profiles, accepted_mapping, dropped_mapping)

    grouped_ids: dict[str, list[int]] = defaultdict(list)
    source_ids_by_target: dict[str, set[str]] = defaultdict(set)
    labels_by_target: dict[str, set[str]] = defaultdict(set)
    datasets_by_target: dict[str, set[str]] = defaultdict(set)
    display_by_target: dict[str, Counter] = defaultdict(Counter)
    counts_by_dataset: Counter = Counter()
    counts_by_valid_status: Counter = Counter()
    counts_by_invalid_reason: Counter = Counter()
    counts_by_risk_flag: Counter = Counter()
    dropped_row_counts: Counter = Counter()

    eligible_row_id = 0
    total_rows = 0
    with gzip.open(args.manifest, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            source_id = row["aphia_id"]
            if source_id in dropped_mapping:
                dropped_row_counts[source_id] += 1
                continue
            target_id = mapped_target_id(source_id, accepted_mapping)
            grouped_ids[target_id].append(eligible_row_id)
            source_ids_by_target[target_id].add(source_id)
            labels_by_target[target_id].add(row["label_original"])
            datasets_by_target[target_id].add(row["dataset_id"])
            display_by_target[target_id][row["class_display_name"]] += 1
            counts_by_dataset[row["dataset_id"]] += 1
            counts_by_valid_status[row["valid_status"]] += 1
            counts_by_invalid_reason[row["invalid_reason"]] += 1
            counts_by_risk_flag[row["risk_flag"]] += 1
            eligible_row_id += 1

    assignments, per_target_split_counts, fallback_counts = split_row_ids(grouped_ids, args.seed, ratios)

    class_rows: list[dict[str, Any]] = []
    for class_index, target_id in enumerate(sorted(grouped_ids, key=lambda value: (not value.isdigit(), value))):
        source_ids = source_ids_by_target[target_id]
        profile = profiles.get(target_id, {})
        display_name = profile.get("class_display_name")
        if not display_name:
            display_name = display_by_target[target_id].most_common(1)[0][0]
        class_rows.append(
            {
                "class_index": class_index,
                "target_id": target_id,
                "target_display_name": display_name,
                "target_kind": classify_target_kind(target_id, source_ids, profiles),
                "source_ids": "|".join(sorted(source_ids)),
                "source_labels": "|".join(sorted(labels_by_target[target_id])),
                "source_datasets": "|".join(sorted(datasets_by_target[target_id])),
                "post_mapping_image_count": len(grouped_ids[target_id]),
                "train_count": per_target_split_counts[target_id]["train"],
                "val_count": per_target_split_counts[target_id]["val"],
                "test_count": per_target_split_counts[target_id]["test"],
                "allocation_rule": per_target_split_counts[target_id]["allocation_rule"],
            }
        )
    target_to_class = {row["target_id"]: int(row["class_index"]) for row in class_rows}

    class_index_path = args.output_dir / "class_index.csv"
    with class_index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    (args.output_dir / "classes.txt").write_text(
        "".join(f"{row['target_display_name']}\n" for row in class_rows)
    )
    (args.output_dir / "target_ids.txt").write_text("".join(f"{row['target_id']}\n" for row in class_rows))
    (args.output_dir / "aphia_ids.txt").write_text("".join(f"{row['target_id']}\n" for row in class_rows))

    split_counts: Counter = Counter()
    split_manifest_fields = [
        "image_path",
        "dataset_id",
        "valid_status",
        "invalid_reason",
        "risk_flag",
        "source_id",
        "target_id",
        "class_index",
        "class_display_name",
        "label_original",
        "split",
        "split_policy",
        "split_seed",
        "width",
        "height",
    ]
    split_files = {
        "train": (args.output_dir / "train.txt").open("w"),
        "val": (args.output_dir / "val.txt").open("w"),
        "test": (args.output_dir / "test.txt").open("w"),
    }
    try:
        with (args.output_dir / "split_manifest.csv").open("w", newline="") as split_handle:
            writer = csv.DictWriter(split_handle, fieldnames=split_manifest_fields)
            writer.writeheader()
            eligible_row_id = 0
            with gzip.open(args.manifest, "rt", newline="") as manifest_handle:
                reader = csv.DictReader(manifest_handle)
                for row in reader:
                    source_id = row["aphia_id"]
                    if source_id in dropped_mapping:
                        continue
                    target_id = mapped_target_id(source_id, accepted_mapping)
                    split = assignments[eligible_row_id]
                    class_index = target_to_class[target_id]
                    split_counts[split] += 1
                    split_files[split].write(f"{row['image_path']} {class_index}\n")
                    writer.writerow(
                        {
                            "image_path": row["image_path"],
                            "dataset_id": row["dataset_id"],
                            "valid_status": row["valid_status"],
                            "invalid_reason": row["invalid_reason"],
                            "risk_flag": row["risk_flag"],
                            "source_id": source_id,
                            "target_id": target_id,
                            "class_index": class_index,
                            "class_display_name": row["class_display_name"],
                            "label_original": row["label_original"],
                            "split": split,
                            "split_policy": "target_stratified_configured_ratios_floor_with_train_min_fallback",
                            "split_seed": args.seed,
                            "width": row["width"],
                            "height": row["height"],
                        }
                    )
                    eligible_row_id += 1
    finally:
        for handle in split_files.values():
            handle.close()

    dropped_rows = []
    for source_id in sorted(dropped_mapping):
        row = dropped_mapping[source_id]
        dropped_rows.append(
            {
                "source_id": source_id,
                "source_display_name": row["source_class_display_name"],
                "mapping_outcome": "dropped",
                "image_count_from_mapping": int(row["source_image_count"]),
                "dropped_manifest_rows": dropped_row_counts[source_id],
                "notes": row.get("notes", ""),
            }
        )

    split_policy = {
        "phase": TAIPEI_PHASE,
        "round": ROUND,
        "stage": STAGE,
        "line_code": LINE_CODE,
        "split_ratio": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "split_unit": "image_row",
        "stratification_key": "post_mapping_target_id",
        "dataset_aware_stratification": False,
        "seed": args.seed,
        "shuffle_scope": "within_each_post_mapping_target",
        "sorting_rule": "manifest_order_for_split_manifest; split txt follows manifest order",
        "allocation_rule": "phase2_floor_allocation_with_minimum_one_train_example_fallback",
        "fallback_counts": dict(fallback_counts),
        "per_target_split_counts": per_target_split_counts,
    }
    write_json(args.output_dir / "split_policy.json", split_policy)

    data_policy = {
        "phase": TAIPEI_PHASE,
        "round": ROUND,
        "stage": STAGE,
        "line_code": LINE_CODE,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "target_profiles_path": str(args.target_profiles),
        "before_training_mapping_path": str(args.before_training_mapping),
        "all_manifest_rows_trainable_under_round_policy": True,
        "valid_status_is_filter": False,
        "invalid_reason_is_filter": False,
        "risk_flag_is_filter": False,
        "mapping_outcome_policy": "accepted and merged rows are kept; dropped rows are excluded",
        "mapping_outcome_counts": dict(mapping_outcome_counts),
        "dropped_mapping_rows": dropped_rows,
        "target_count_before_mapping": len(profiles),
        "target_count_after_mapping": len(class_rows),
        "total_manifest_rows": total_rows,
        "eligible_rows_after_mapping": sum(split_counts.values()),
        "split_counts": dict(split_counts),
        "dataset_counts": dict(counts_by_dataset),
        "valid_status_counts": dict(counts_by_valid_status),
        "invalid_reason_counts": dict(counts_by_invalid_reason),
        "risk_flag_counts": dict(counts_by_risk_flag),
        "class_weight_policy": {
            "mode": "panaiotis_relative",
            "exponent": 0.22,
            "basis": "train split class counts",
        },
    }
    write_json(args.output_dir / "data_policy.json", data_policy)

    checksums = {
        name: sha256_file(args.output_dir / name)
        for name in [
            "target_id_map.csv",
            "class_index.csv",
            "split_manifest.csv",
            "split_policy.json",
            "data_policy.json",
            "train.txt",
            "val.txt",
            "test.txt",
            "classes.txt",
            "target_ids.txt",
            "aphia_ids.txt",
        ]
    }
    source_refs = {
        "phase": TAIPEI_PHASE,
        "round": ROUND,
        "stage": STAGE,
        "line_code": LINE_CODE,
        "manifest": {"path": str(args.manifest), "sha256": manifest_hash},
        "target_profiles": {
            "path": str(args.target_profiles),
            "sha256": sha256_file(args.target_profiles),
        },
        "before_training_mapping": {
            "path": str(args.before_training_mapping),
            "sha256": sha256_file(args.before_training_mapping),
        },
        "supporting_data_dir": str(args.output_dir),
        "supporting_data_sha256": checksums,
        "summary": {
            "target_count_before_mapping": len(profiles),
            "target_count_after_mapping": len(class_rows),
            "eligible_rows_after_mapping": sum(split_counts.values()),
            "split_counts": dict(split_counts),
            "mapping_outcome_counts": dict(mapping_outcome_counts),
            "dropped_source_targets": dropped_rows,
        },
    }
    write_json(args.output_dir / "source_refs.json", source_refs)

    print(json.dumps(source_refs["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
