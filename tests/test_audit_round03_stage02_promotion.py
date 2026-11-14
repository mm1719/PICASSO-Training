from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_round03_stage02_promotion import markdown_table
from audit_round03_stage02_promotion import score_confusion
from audit_round03_stage02_promotion import high_support_table


class PromotionAuditTest(unittest.TestCase):
    def test_score_confusion_uses_macro_f1_and_balanced_accuracy(self) -> None:
        matrix = np.array([[8, 2], [1, 9]])
        macro_f1, balanced_accuracy, recall, f1 = score_confusion(matrix)

        self.assertAlmostEqual(balanced_accuracy, 0.85)
        self.assertAlmostEqual(recall[0], 0.8)
        self.assertAlmostEqual(recall[1], 0.9)
        self.assertAlmostEqual(macro_f1, f1.mean())

    def test_markdown_table_does_not_require_tabulate(self) -> None:
        import pandas as pd

        table = markdown_table(pd.DataFrame({"metric": ["macro_f1"], "delta": [0.01]}))
        self.assertIn("| metric | delta |", table)
        self.assertIn("| macro_f1 | 0.010000 |", table)

    def test_high_support_table_keeps_only_flagged_classes(self) -> None:
        import pandas as pd

        classwise = pd.DataFrame(
            {
                "true_class_name": ["common", "rare"],
                "validation_support": [5000, 12],
                "high_support": [True, False],
                "recall_delta": [0.01, -0.1],
                "recall_ci95_low": [0.0, -0.3],
                "recall_ci95_high": [0.02, 0.1],
                "f1_delta": [0.02, -0.2],
                "f1_ci95_low": [0.01, -0.4],
                "f1_ci95_high": [0.03, 0.0],
            }
        )

        table = high_support_table(classwise)

        self.assertEqual(table["true_class_name"].tolist(), ["common"])


if __name__ == "__main__":
    unittest.main()
