import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from aofs.datasets import get_dataset_spec
from aofs.obb_eval import aggregate_group_aps, count_ground_truths, evaluate_detection_files


class ObbEvaluationTest(unittest.TestCase):
    def test_cli_help_does_not_require_numeric_or_polygon_extensions(self):
        completed = subprocess.run(
            [sys.executable, "DOTA_devkit/dota_evaluation_task1.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--dataset", completed.stdout)

    def test_voc_eval_uses_package_aware_polygon_extension_import(self):
        source = Path("DOTA_devkit/dota_evaluation_task1.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from DOTA_devkit import polyiou", source)

    def test_legacy_cli_requires_explicit_dataset_and_uses_shared_evaluator(self):
        source = Path("DOTA_devkit/dota_evaluation_task1.py").read_text(encoding="utf-8")
        self.assertIn("parser.add_argument('--dataset'", source)
        self.assertIn("evaluate_obb_predictions(", source)

    def test_nwpu_registry_uses_ten_classes_and_expected_novel_split(self):
        spec = get_dataset_spec("nwpu")
        self.assertEqual(len(spec.classes), 10)
        self.assertEqual(spec.novel, ("airplane", "baseball-diamond", "tennis-court"))
        self.assertEqual(set(spec.classes), set(spec.novel) | set(spec.base))

    def test_missing_detection_class_contributes_zero_to_denominator(self):
        metric = aggregate_group_aps(
            {"airplane": 0.6, "baseball-diamond": 0.3},
            ("airplane", "baseball-diamond", "tennis-court"),
        )
        self.assertAlmostEqual(metric, 0.3)

    def test_evaluator_calls_voc_eval_only_for_nonempty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Task1_airplane.txt").write_text(
                "img 0.9 0 0 1 0 1 1 0 1\n", encoding="utf-8"
            )
            (root / "Task1_ship.txt").write_text("", encoding="utf-8")
            calls = []

            def fake_voc_eval(detpath, annopath, imagesetfile, classname, ovthresh, use_07_metric):
                calls.append(classname)
                return [], [], 0.5

            result = evaluate_detection_files(
                detection_dir=root,
                annopath="annotations/{:s}.txt",
                imagesetfile="images.txt",
                dataset_name="nwpu",
                voc_eval_fn=fake_voc_eval,
            )
            self.assertEqual(calls, ["airplane"])
            self.assertEqual(result["per_class"]["tennis-court"], 0.0)
            self.assertEqual(result["prediction_counts"]["airplane"], 1)
            self.assertEqual(result["prediction_counts"]["tennis-court"], 0)
            self.assertAlmostEqual(result["novel_map50"], 0.5 / 3)

    def test_ground_truth_counts_exclude_difficult_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images.txt").write_text("scene_a\n", encoding="utf-8")
            (root / "scene_a.txt").write_text(
                "0 0 1 0 1 1 0 1 airplane 0\n"
                "0 0 1 0 1 1 0 1 airplane 1\n"
                "0 0 1 0 1 1 0 1 ship 0\n",
                encoding="utf-8",
            )
            counts = count_ground_truths(
                str(root / "{:s}.txt"), root / "images.txt", "nwpu"
            )
            self.assertEqual(counts["airplane"], 1)
            self.assertEqual(counts["ship"], 1)
            self.assertEqual(counts["tennis-court"], 0)


if __name__ == "__main__":
    unittest.main()
