import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aofs.datasets import get_dataset_spec
from aofs.fewshot import Candidate, generate_split, select_candidates


class FewShotSelectionTest(unittest.TestCase):
    def test_cli_help_exposes_profile_seed_and_portable_roots(self):
        completed = subprocess.run(
            [sys.executable, "tools/gen_fewlist_nwpu.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--data-root", completed.stdout)
        self.assertIn("--output-root", completed.stdout)
        self.assertIn("--profile", completed.stdout)
        self.assertIn("--seed", completed.stdout)

    def setUp(self):
        self.candidates = [
            Candidate("scene_a.png", "airplane", (str(index),) * 8, "0")
            for index in range(4)
        ] + [
            Candidate(f"scene_{letter}.png", "airplane", (letter,) * 8, "0")
            for letter in ("b", "c", "d")
        ]

    def test_paper_selection_is_exact_and_reproducible(self):
        first = select_candidates(self.candidates, shot=3, seed=7, max_per_image=None)
        second = select_candidates(self.candidates, shot=3, seed=7, max_per_image=None)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_robust_selection_caps_one_instance_per_image(self):
        selected = select_candidates(self.candidates, shot=3, seed=7, max_per_image=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({item.image_path for item in selected}), 3)

    def test_insufficient_distinct_images_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "only 1 eligible images"):
            select_candidates(self.candidates[:4], shot=3, seed=7, max_per_image=1)

    def test_generate_split_writes_all_classes_and_deterministic_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            image_dir = root / "training" / "images"
            label_dir = root / "training" / "labelTxt"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image_paths = []
            for class_name in get_dataset_spec("nwpu").classes:
                image_path = image_dir / f"{class_name}.png"
                image_path.write_bytes(b"")
                image_paths.append(image_path)
                (label_dir / f"{class_name}.txt").write_text(
                    f"0 0 10 0 10 10 0 10 {class_name} 0\n",
                    encoding="utf-8",
                )
            (root / "training.txt").write_text(
                "\n".join(str(path) for path in image_paths) + "\n",
                encoding="utf-8",
            )

            first = generate_split(root, Path(directory) / "split_a", "paper", 1, 11, None)
            second = generate_split(root, Path(directory) / "split_b", "paper", 1, 11, None)
            first_audit = json.loads(first["audit"].read_text(encoding="utf-8"))
            second_audit = json.loads(second["audit"].read_text(encoding="utf-8"))
            self.assertEqual(first_audit["manifest_sha256"], second_audit["manifest_sha256"])
            self.assertEqual(len(first["meta"].read_text(encoding="utf-8").splitlines()), 10)
            self.assertTrue((first["support_root"] / "airplane" / "airplane.txt").is_file())


if __name__ == "__main__":
    unittest.main()
