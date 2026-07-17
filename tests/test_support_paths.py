import os
import unittest
from pathlib import Path

from aofs.experiment import support_label_path


class SupportPathTest(unittest.TestCase):
    def test_cfg_prefers_explicit_shot_and_support_root(self):
        source = Path("cfg.py").read_text(encoding="utf-8")
        self.assertIn("__C.shot = int(dataopt['shot'])", source)
        self.assertIn("__C.support_root", source)

    def test_meta_dataset_uses_generated_support_root(self):
        source = Path("dataset.py").read_text(encoding="utf-8")
        self.assertIn("support_label_path(cfg.support_root", source)

    def test_train_exposes_reproducibility_overrides(self):
        source = Path("train.py").read_text(encoding="utf-8")
        for argument in (
            "--profile",
            "--stage",
            "--dataset-name",
            "--shot",
            "--seed",
            "--data-root",
            "--meta",
            "--support-root",
        ):
            self.assertIn(f"parser.add_argument('{argument}'", source)

    def test_support_root_separates_seed_specific_labels(self):
        result = support_label_path(
            "/splits/seed7/support_labels",
            "airplane",
            "/data/training/images/001.png",
        )
        self.assertEqual(
            result,
            os.path.normpath("/splits/seed7/support_labels/airplane/001.txt"),
        )


if __name__ == "__main__":
    unittest.main()
