from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.phase4_outputs.common import ensure_dirs
from scripts.phase4_outputs.metrics import export_training_epoch_metrics


class TrainingMetricExportTest(unittest.TestCase):
    def test_exports_optional_validation_metrics_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            logs = run_dir / "logs"
            logs.mkdir(parents=True)
            (logs / "epoch_metrics.csv").write_text(
                "\n".join(
                    [
                        "epoch,train_loss,train_accuracy,val_loss,val_accuracy,"
                        "val_balanced_accuracy,val_macro_f1,val_weighted_f1,"
                        "raw_val_loss,raw_val_accuracy,raw_val_balanced_accuracy,"
                        "raw_val_macro_f1,raw_val_weighted_f1,lr,seconds",
                        "1,0.9,0.8,0.7,0.91,0.55,0.60,0.92,0.71,0.90,0.54,0.59,0.91,0.001,12.3",
                    ]
                )
                + "\n"
            )
            dirs = ensure_dirs(root / "out")

            exported = export_training_epoch_metrics(run_dir, dirs)

            self.assertEqual(float(exported.loc[0, "val_balanced_accuracy"]), 0.55)
            self.assertEqual(float(exported.loc[0, "val_macro_f1"]), 0.60)
            self.assertEqual(float(exported.loc[0, "raw_val_weighted_f1"]), 0.91)


if __name__ == "__main__":
    unittest.main()
