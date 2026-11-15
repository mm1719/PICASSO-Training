from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_round00_stage01_timm import predict_test


class PredictionRankingTest(unittest.TestCase):
    def assert_prediction_contract(self, logits, *, multicrop=False):
        count, class_count = logits.shape
        labels = torch.arange(count) % class_count
        names = [f"synthetic_{i}" for i in range(class_count)]
        targets = [str(i) for i in range(class_count)]
        batches = []
        for start, end in ((0, 7), (7, count)):
            values = logits[start:end]
            if multicrop:
                values = values[:, None, :, None, None].expand(-1, 2, -1, 1, 1)
            metadata = {
                "image_path": [f"synthetic/{i}.png" for i in range(start, end)],
                "dataset_id": ["synthetic"] * (end - start),
                "valid_status": ["valid"] * (end - start),
                "invalid_reason": ["n/a"] * (end - start),
                "risk_flag": ["n/a"] * (end - start),
                "source_id": [str(i) for i in range(start, end)],
                "target_id": [targets[i] for i in labels[start:end].tolist()],
                "split": ["test"] * (end - start),
            }
            batches.append((values, labels[start:end], metadata))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.csv"
            metrics = predict_test(
                torch.nn.Flatten(start_dim=1),
                batches,
                torch.device("cpu"),
                names,
                targets,
                False,
                output,
            )
            frame = pd.read_csv(output, float_precision="round_trip")
        ranking = frame[[f"pred_top{k}_index" for k in range(1, 6)]].to_numpy()
        expected = logits.topk(5, dim=1).indices
        np.testing.assert_array_equal(ranking, expected.numpy())
        probabilities = torch.softmax(logits, dim=1).gather(1, expected)
        np.testing.assert_array_equal(
            frame[[f"pred_top{k}_prob" for k in range(1, 6)]].to_numpy(),
            probabilities.double().numpy(),
        )
        self.assertEqual(metrics["num_samples"], count)
        for k in range(1, 6):
            hits = (ranking[:, :k] == labels.numpy()[:, None]).any(axis=1)
            self.assertEqual(metrics["topk_accuracy"][f"top{k}_accuracy"], hits.mean())
        self.assertEqual(
            frame["pred_top1_name"].tolist(), [names[i] for i in ranking[:, 0]]
        )
        self.assertEqual(
            frame["pred_top1_target_id"].tolist(), ranking[:, 0].tolist()
        )

    def test_fp16_softmax_underflow_keeps_logit_ranking(self):
        logits = torch.eye(20, dtype=torch.float16) * 5
        logits[1] = torch.tensor([20, *range(1, -18, -1)], dtype=torch.float16)
        self.assertEqual(torch.count_nonzero(logits[1].softmax(0)[1:]).item(), 0)
        self.assert_prediction_contract(logits)

    def test_fp16_softmax_rounding_keeps_distinct_logit_ranking(self):
        logits = torch.eye(20, dtype=torch.float16) * 5
        logits[1] = torch.tensor([0, 1e-6, *range(-1, -19, -1)])
        self.assertGreater(logits[1, 1].item(), logits[1, 0].item())
        self.assertEqual(logits[1].softmax(0)[0], logits[1].softmax(0)[1])
        self.assert_prediction_contract(logits)

    def test_exact_logit_ties_keep_metrics_and_csv_consistent(self):
        self.assert_prediction_contract(torch.zeros(20, 20, dtype=torch.float16))

    def test_fp32_multibatch_scores_and_labels_remain_aligned(self):
        base = torch.linspace(2, -2, 20)
        self.assert_prediction_contract(torch.stack([base.roll(i) for i in range(20)]))

    def test_multicrop_scores_and_labels_remain_aligned(self):
        base = torch.linspace(2, -2, 20)
        self.assert_prediction_contract(
            torch.stack([base.roll(i) for i in range(20)]), multicrop=True
        )


if __name__ == "__main__":
    unittest.main()
