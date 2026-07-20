import os
import unittest
from unittest.mock import patch

from aofs.experiment import (
    build_experiment_identity,
    compute_tuning_epochs,
    override_dataset_root,
    resolve_data_options,
    should_validate,
    validate_resume_identity,
)


class ExperimentTest(unittest.TestCase):
    def test_dataset_yaml_root_can_be_overridden_without_mutating_source(self):
        source = {"path": "/workspace/dataset", "train": "training/images"}
        resolved = override_dataset_root(source, "/srv/nwpu")
        self.assertEqual(resolved["path"], os.path.normpath("/srv/nwpu"))
        self.assertEqual(source["path"], "/workspace/dataset")

    def test_resolve_data_options_uses_cli_root_and_expands_environment(self):
        options = {
            "train": "${AOFS_DATA_ROOT}/training.txt",
            "valid": "${AOFS_DATA_ROOT}/evaluation.txt",
            "meta": "${AOFS_SPLIT_ROOT}/meta.txt",
        }
        with patch.dict(os.environ, {"AOFS_SPLIT_ROOT": "/runs/split"}, clear=False):
            resolved = resolve_data_options(options, data_root="/data/nwpu")
        self.assertEqual(resolved["train"], os.path.normpath("/data/nwpu/training.txt"))
        self.assertEqual(resolved["valid"], os.path.normpath("/data/nwpu/evaluation.txt"))
        self.assertEqual(resolved["meta"], os.path.normpath("/runs/split/meta.txt"))

    def test_unresolved_environment_variable_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "AOFS_SPLIT_ROOT"):
                resolve_data_options({"meta": "${AOFS_SPLIT_ROOT}/meta.txt"})

    def test_non_path_profile_values_expand_environment_variables(self):
        with patch.dict(os.environ, {"AOFS_SEED": "2"}, clear=False):
            resolved = resolve_data_options({"seed": "${AOFS_SEED}", "profile": "paper"})
        self.assertEqual(resolved["seed"], "2")

    def test_tuning_epochs_match_author_conversion(self):
        self.assertEqual(compute_tuning_epochs(max_epoch=50000, repeat=100), 500)

    def test_noval_still_validates_final_epoch(self):
        self.assertFalse(should_validate(epoch=98, epochs=100, noval=True, period=10))
        self.assertTrue(should_validate(epoch=99, epochs=100, noval=True, period=10))

    def test_periodic_validation_includes_period_and_final_epoch(self):
        self.assertTrue(should_validate(epoch=9, epochs=100, noval=False, period=10))
        self.assertTrue(should_validate(epoch=99, epochs=100, noval=False, period=10))

    def test_resume_rejects_different_shot_or_seed(self):
        expected = build_experiment_identity("paper", "nwpu", 5, 1, "fewtune")
        actual = build_experiment_identity("paper", "nwpu", 3, 1, "fewtune")
        with self.assertRaisesRegex(ValueError, "shot"):
            validate_resume_identity(expected, actual)


if __name__ == "__main__":
    unittest.main()
