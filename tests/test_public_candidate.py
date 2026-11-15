from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

from scripts import audit_round03_stage02_promotion as audit
from scripts import prepare_round00_stage01_data as prepare
from scripts.evaluate_predictions import evaluate_predictions
from scripts.synthetic_demo import prediction_frame, run_demo
from scripts.validate_config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def trainer_functions(*names: str) -> dict:
    """Run selected definitions only, with framework calls mocked, never import torch."""
    tree = ast.parse((ROOT / "scripts/train_round00_stage01_timm.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Trainer definition selection is incomplete")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    fake_torch = SimpleNamespace(tensor=lambda values, dtype: np.array(values, dtype=dtype), float32=np.float32, double=np.float64, load=MagicMock(), compile=MagicMock())
    namespace = {"__builtins__": __builtins__, "np": np, "torch": fake_torch, "LOGGER": logging.getLogger("test"), "Path": Path, "hashlib": hashlib, "validate_config": validate_config}
    exec(compile(ast.fix_missing_locations(module), "<selected-trainer-definitions>", "exec"), namespace)
    return namespace


class SyntheticPipelineTest(unittest.TestCase):
    def test_end_to_end_contracts_and_identical_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "demo"
            result = run_demo(out)
            self.assertTrue(result["synthetic_only"])
            self.assertEqual(result["split_counts"], {"train": 80, "test": 10, "val": 10})
            self.assertFalse(result["model_created"])
            self.assertFalse(result["images_created_or_opened"])
            summary = json.loads((out / "paired_audit/summary.json").read_text())
            for row in summary["global_metrics"]:
                self.assertEqual(row["delta"], 0)
                self.assertEqual(row["bootstrap_ci95_low"], 0)
                self.assertEqual(row["bootstrap_ci95_high"], 0)
            policy = json.loads((out / "supporting/data_policy.json").read_text())
            self.assertFalse(policy["valid_status_is_filter"])
            self.assertEqual(policy["eligible_rows_after_mapping"], 100)
            self.assertEqual(policy["valid_status_counts"], {"valid": 50, "invalid": 50})
            metrics = json.loads((out / "evaluation/metrics.json").read_text())
            self.assertAlmostEqual(metrics["topk_accuracy"]["top1_accuracy"], 0.6)
            self.assertEqual(metrics["topk_accuracy"]["top5_accuracy"], 1.0)
            self.assertEqual(metrics["support_basis"], "post-mapping training split count")
            self.assertTrue((out / "evaluation/csv/richer_slices/classification_report_by_dataset.csv").exists())
            self.assertNotIn("point estimates favor A2", (out / "paired_audit/report.md").read_text())
            self.assertFalse(any(p.suffix in {".png", ".pt", ".jpg"} for p in out.rglob("*")))
            with self.assertRaises(FileExistsError):
                run_demo(out)

    def test_stratification_is_seeded_and_small_classes_keep_training(self):
        groups = {"b": list(range(10)), "a": [10]}
        first = prepare.split_row_ids(groups, 42, (0.8, 0.1, 0.1))
        self.assertEqual(first, prepare.split_row_ids(groups, 42, (0.8, 0.1, 0.1)))
        self.assertEqual(first[0][10], "train")
        self.assertEqual(len(first[0]), 11)
        self.assertEqual(groups["b"], list(range(10)))

    def test_rejects_bad_ratio_before_reading_inputs(self):
        with patch.object(sys, "argv", ["prepare", "--manifest", "absent", "--target-profiles", "absent", "--before-training-mapping", "absent", "--output-dir", "unused", "--expected-manifest-sha256", "unused", "--train-ratio", "-1", "--val-ratio", "1", "--test-ratio", "1"]):
            with self.assertRaisesRegex(ValueError, "Split ratios"):
                prepare.main()

    def assert_manifest_checksum_rejected(self, expected_checksum):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            fixture = root / "fixture"
            run_demo(fixture)
            inputs = fixture / "synthetic_inputs"
            output = root / "rejected"
            args = [
                "prepare",
                "--manifest", str(inputs / "manifest.csv.gz"),
                "--target-profiles", str(inputs / "profiles.csv"),
                "--before-training-mapping", str(inputs / "mapping.csv"),
                "--output-dir", str(output),
                "--expected-manifest-sha256", expected_checksum,
            ]
            with patch.object(sys, "argv", args):
                with self.assertRaisesRegex(ValueError, "Manifest checksum mismatch"):
                    prepare.main()
            self.assertFalse(output.exists())

    def test_rejects_empty_manifest_checksum_without_creating_output(self):
        self.assert_manifest_checksum_rejected("")

    def test_rejects_wrong_manifest_checksum_without_creating_output(self):
        self.assert_manifest_checksum_rejected("0" * 64)


class AuditValidationTest(unittest.TestCase):
    def frame(self):
        return prediction_frame(pd.DataFrame([{"image_path": f"synthetic/{i}", "dataset_id": "fake", "source_id": str(i), "target_id": str(i), "class_index": i, "split": "val"} for i in range(5)]))

    def load(self, left, right):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            left.to_csv(a, index=False)
            right.to_csv(b, index=False)
            return audit.load_paired_predictions(a, b)

    def test_accepts_exact_alignment(self):
        self.assertEqual(len(self.load(self.frame(), self.frame())[0]), 5)

    def test_rejects_misalignment_test_split_duplicates_and_empty(self):
        good = self.frame()
        for bad in [good.iloc[::-1], good.assign(split="test"), pd.concat([good, good.iloc[[0]]]), good.iloc[:0]]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                self.load(bad, bad if len(bad) != 5 or bad["split"].iloc[0] == "test" else good)

    def test_rejects_noncontiguous_fractional_and_out_of_range_indexes(self):
        good = self.frame()
        for bad in [good.assign(true_class_index=[0, 1, 2, 3, 5]), good.assign(pred_top1_index=0.2), good.assign(pred_top1_index=10)]:
            with self.assertRaises(ValueError):
                self.load(bad, bad)

    def test_bootstrap_is_paired_and_reproducible(self):
        true = np.repeat(np.arange(2), 5)
        baseline = np.zeros(10, dtype=int)
        candidate = true.copy()
        first = audit.bootstrap_deltas(true, baseline, candidate, 2, 30, 123)
        second = audit.bootstrap_deltas(true, baseline, candidate, 2, 30, 123)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertTrue((first[0] > 0).all())


class EvaluationValidationTest(unittest.TestCase):
    def test_numeric_string_training_counts_are_normalized(self):
        source = pd.DataFrame([{"image_path": f"synthetic/{i}", "dataset_id": "fake", "valid_status": "valid", "invalid_reason": "n/a", "risk_flag": "n/a", "class_index": i} for i in range(5)])
        classes = pd.DataFrame({"class_index": range(5), "target_display_name": [f"class{i}" for i in range(5)], "train_count": ["10"]*5})
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_predictions(prediction_frame(source), classes, Path(tmp) / "out")
        self.assertEqual(result["sample_count"], 5)

    def test_rejects_invalid_rankings_without_creating_output(self):
        source = pd.DataFrame([{"image_path": f"synthetic/{i}", "dataset_id": "fake", "valid_status": "valid", "invalid_reason": "n/a", "risk_flag": "n/a", "class_index": i} for i in range(5)])
        frame = prediction_frame(source)
        classes = pd.DataFrame({"class_index": range(5), "target_display_name": [f"class{i}" for i in range(5)], "train_count": [10]*5})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            for bad in [frame.assign(pred_top1_prob=1.2), frame.assign(pred_top1_index=99), frame.assign(pred_top1_index=0.2), frame.assign(pred_top2_index=frame["pred_top1_index"]), frame.iloc[:0]]:
                with self.assertRaises(ValueError):
                    evaluate_predictions(bad, classes, output)
                self.assertFalse(output.exists())


class TrainerMockTest(unittest.TestCase):
    def test_static_safety_gate_and_config_only_dry_run(self):
        conf = yaml.safe_load((ROOT / "configs/training.example.yaml").read_text())
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_config(conf)
        conf["manifest"]["sha256"] = "a" * 64
        validate_config(conf)
        for section, key, value in [("model", "pretrained", True), ("model", "pretrained_checkpoint", "anything"), ("monitor", "use_wandb", True), ("model", "num_classes", 3), ("model", "modelname", "hf-hub:remote/model"), ("model", "extra_kwargs", {"checkpoint_path": "anything"})]:
            altered = copy.deepcopy(conf)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(altered)
        ns = trainer_functions("main")
        ns["parse_args"] = lambda: SimpleNamespace(config=Path("synthetic.yaml"), dry_run=True)
        ns["load_yaml"] = lambda path: conf
        with contextlib.redirect_stdout(io.StringIO()):
            ns["main"]()
        ns["torch"].load.assert_not_called()
        ns["torch"].compile.assert_not_called()
        ns["parse_args"] = lambda: SimpleNamespace(config=Path("synthetic.yaml"), dry_run=False, allow_local_training=False)
        with self.assertRaisesRegex(ValueError, "opt-in"):
            ns["main"]()

    def test_sampler_caps_large_classes_with_fake_tensor_backend(self):
        ns = trainer_functions("class_count_cap_sample_weights", "compute_class_weights")
        rows = [SimpleNamespace(class_index=0)] * 10 + [SimpleNamespace(class_index=1)] * 5 + [SimpleNamespace(class_index=2)]
        weights = ns["class_count_cap_sample_weights"](rows, 3, 5)
        self.assertEqual(weights.tolist(), [0.5] * 10 + [1.0] * 6)
        with self.assertRaises(ValueError):
            ns["class_count_cap_sample_weights"](rows, 3, 0)
        with self.assertRaises(ValueError):
            ns["compute_class_weights"](rows, 4, "balanced", 0.22)

    def test_compile_arguments_and_resume_checks_are_mock_only(self):
        ns = trainer_functions("maybe_compile_model", "load_last_checkpoint")
        dummy = object()
        ns["maybe_compile_model"](dummy, {"training": {"torch_compile": {"enabled": True, "dynamic": False, "options": {"triton.cudagraphs": True}}}})
        ns["torch"].compile.assert_called_once_with(dummy, dynamic=False, options={"triton.cudagraphs": True})
        with self.assertRaises(ValueError):
            ns["maybe_compile_model"](dummy, {"training": {"torch_compile": {"enabled": True, "mode": "default", "options": {"x": 1}}}})
        ns["torch"].load.return_value = {"checkpoint_kind": "last", "config_sha256": "old", "num_classes": 3}
        with self.assertRaisesRegex(ValueError, "Config hash mismatch"):
            ns["load_last_checkpoint"](Path("not-a-real-checkpoint"), expected_config_sha256="new", expected_num_classes=3)
        with self.assertRaisesRegex(ValueError, "num_classes mismatch"):
            ns["load_last_checkpoint"](Path("not-a-real-checkpoint"), expected_config_sha256="old", expected_num_classes=4)

    def test_no_credentials_import_or_tracking_initializer(self):
        text = (ROOT / "scripts/train_round00_stage01_timm.py").read_text()
        self.assertNotIn("import netrc", text)
        self.assertNotIn("wandb.init(", text)
        self.assertNotIn("WANDB_API_KEY", text)


if __name__ == "__main__":
    unittest.main()
