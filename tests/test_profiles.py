import unittest
from pathlib import Path

def read_simple_yaml(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        values[key] = float(value)
    return values


def read_data_cfg(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        values[key] = value
    return values


class ProfileTest(unittest.TestCase):
    def test_paper_profiles_have_full_training_length_and_explicit_identity(self):
        for shot in (3, 5, 10):
            values = read_data_cfg(Path(f"cfg/paper/nwpu_{shot}shot.data"))
            self.assertEqual(values["profile"], "paper")
            self.assertEqual(int(values["shot"]), shot)
            self.assertEqual(int(values["max_epoch"]), 50000)
            self.assertEqual(int(values["repeat"]), 100)
            self.assertEqual(values["seed"], "${AOFS_SEED}")
            self.assertEqual(values["meta"], "${AOFS_SPLIT_ROOT}/meta.txt")
            self.assertEqual(
                values["support_root"], "${AOFS_SPLIT_ROOT}/support_labels"
            )

    def test_robust_profiles_keep_training_budget_but_have_distinct_identity(self):
        for shot in (3, 5, 10):
            values = read_data_cfg(Path(f"cfg/robust/nwpu_{shot}shot.data"))
            self.assertEqual(values["profile"], "robust")
            self.assertEqual(int(values["shot"]), shot)
            self.assertEqual(int(values["max_epoch"]), 50000)
            self.assertEqual(int(values["repeat"]), 100)

    def test_base_hyperparameters_match_validated_author_code(self):
        values = read_simple_yaml("cfg/paper/hyp.base_nwpu.yaml")
        self.assertEqual(values["lr0"], 0.001)
        self.assertEqual(values["momentum"], 0.937)
        self.assertEqual(values["weight_decay"], 0.0005)

    def test_profile_hyperparameters_match_the_paper(self):
        for profile in ("paper", "robust"):
            values = read_simple_yaml(f"cfg/{profile}/hyp.finetune_nwpu.yaml")
            self.assertEqual(values["lr0"], 0.001)
            self.assertEqual(values["momentum"], 0.999)
            self.assertEqual(values["weight_decay"], 0.0005)

    def test_profiles_use_environment_paths_and_separate_projects(self):
        paper = Path("scripts/train_nwpu_paper.sh").read_text(encoding="utf-8")
        robust = Path("scripts/train_nwpu_robust.sh").read_text(encoding="utf-8")
        for source in (paper, robust):
            self.assertIn("set -euo pipefail", source)
            self.assertIn("AOFS_DATA_ROOT", source)
            self.assertIn("--checkpoint-metric obb_novel_map50", source)
            self.assertIn("--obb-annopath", source)
            self.assertIn("--obb-imagesetfile", source)
            self.assertIn("--patience 500", source)
            self.assertNotIn("AOFS_OBB_ANNOPATH:-${AOFS_DATA_ROOT}", source)
            self.assertIn(
                'OBB_ANNOPATH="${AOFS_DATA_ROOT}/evaluation/labelTxt/{:s}.txt"',
                source,
            )
        self.assertIn("runs/paper", paper)
        self.assertIn("runs/robust", robust)
        self.assertIn("--profile paper", paper)
        self.assertIn("--profile robust", robust)

    def test_eval_script_runs_json_prediction_then_shared_obb_evaluator(self):
        source = Path("scripts/eval_nwpu_obb.sh").read_text(encoding="utf-8")
        self.assertIn("--prediction-stem", source)
        self.assertIn("--save-json", source)
        self.assertIn("DOTA_devkit/dota_evaluation_task1.py", source)
        self.assertIn("--dataset nwpu", source)
        self.assertNotIn("AOFS_OBB_ANNOPATH:-${AOFS_DATA_ROOT}", source)


if __name__ == "__main__":
    unittest.main()
