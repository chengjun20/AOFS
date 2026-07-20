import json
import tempfile
import unittest
from pathlib import Path

from aofs.results import load_metric_run, summarize_metric_files, summarize_runs


class ResultSummaryTest(unittest.TestCase):
    def test_three_seed_summary_reports_mean_and_sample_std(self):
        summary = summarize_runs(
            [
                {"seed": 0, "novel_map50": 0.50},
                {"seed": 1, "novel_map50": 0.60},
                {"seed": 2, "novel_map50": 0.70},
            ]
        )
        self.assertAlmostEqual(summary["novel_map50_mean"], 0.60)
        self.assertAlmostEqual(summary["novel_map50_std"], 0.10)
        self.assertEqual(summary["seeds"], [0, 1, 2])

    def test_metric_loader_finds_parent_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            metric_dir = run / "obb_epoch_0499"
            metric_dir.mkdir(parents=True)
            (run / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "identity": {
                            "profile": "paper",
                            "dataset": "nwpu",
                            "shot": 5,
                            "seed": 2,
                            "stage": "fewtune",
                        }
                    }
                ),
                encoding="utf-8",
            )
            metrics = metric_dir / "obb_metrics.json"
            metrics.write_text(json.dumps({"novel_map50": 0.64}), encoding="utf-8")
            loaded = load_metric_run(metrics)
            self.assertEqual(loaded["profile"], "paper")
            self.assertEqual(loaded["shot"], 5)
            self.assertEqual(loaded["seed"], 2)
            self.assertEqual(loaded["novel_map50"], 0.64)

    def test_summary_rejects_mixed_profile_or_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed, profile in enumerate(("paper", "paper", "robust")):
                run = Path(directory) / f"run{seed}"
                run.mkdir()
                (run / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "identity": {
                                "profile": profile,
                                "dataset": "nwpu",
                                "shot": 3,
                                "seed": seed,
                                "stage": "fewtune",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                metric = run / "obb_metrics.json"
                metric.write_text(json.dumps({"novel_map50": 0.5}), encoding="utf-8")
                paths.append(metric)
            with self.assertRaisesRegex(ValueError, "profile"):
                summarize_metric_files(paths)


if __name__ == "__main__":
    unittest.main()
