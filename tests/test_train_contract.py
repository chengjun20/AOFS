import re
import unittest
from pathlib import Path


class TrainingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("train.py").read_text(encoding="utf-8")

    def test_training_uses_shared_validation_decision(self):
        self.assertIn("should_validate(", self.source)

    def test_checkpoint_contains_experiment_identity(self):
        self.assertIn("'experiment': experiment_identity", self.source)

    def test_obb_metric_can_select_best_checkpoint(self):
        self.assertIn("obb_metrics['novel_map50']", self.source)
        self.assertIn("best_obb_metrics.json", self.source)

    def test_validation_loader_is_active_code(self):
        self.assertRegex(self.source, re.compile(r"^\s+val_loader = create_dataloader\(", re.MULTILINE))
        self.assertRegex(self.source, re.compile(r"^\s+metaset_val = MetaDataset\(", re.MULTILINE))

    def test_cli_exposes_validation_metric_inputs(self):
        for argument in (
            "--val-period",
            "--checkpoint-metric",
            "--obb-annopath",
            "--obb-imagesetfile",
        ):
            self.assertIn(f"parser.add_argument('{argument}'", self.source)

    def test_training_writes_manifest_and_validates_resume_identity(self):
        self.assertIn("write_run_manifest(", self.source)
        self.assertIn("validate_resume_identity(", self.source)

    def test_new_stage_does_not_restore_previous_optimizer(self):
        self.assertIn("if resume and ckpt['optimizer'] is not None", self.source)

    def test_dataset_yaml_uses_portable_root_override(self):
        self.assertIn("override_dataset_root(", self.source)

    def test_named_profiles_do_not_append_legacy_tuning_suffix_after_increment(self):
        self.assertIn("if cfg.tuning and cfg.profile == 'legacy':", self.source)


if __name__ == "__main__":
    unittest.main()
