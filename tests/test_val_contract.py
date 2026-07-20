import ast
import unittest
from pathlib import Path


def load_val_function(name):
    source = Path("val.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    if function is None:
        raise AssertionError(f"{name} is missing from val.py")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "val.py", "exec"), namespace)
    return namespace[name]


class ValidationContractTest(unittest.TestCase):
    def test_cli_accepts_prediction_stem(self):
        source = Path("val.py").read_text(encoding="utf-8")
        self.assertIn("parser.add_argument('--prediction-stem'", source)

    def test_run_accepts_prediction_stem_and_writes_even_empty_json(self):
        source = Path("val.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        arguments = [argument.arg for argument in run.args.args]
        self.assertIn("prediction_stem", arguments)
        self.assertIn("if save_json:", source)
        self.assertNotIn("if save_json and len(jdict):", source)

    def test_standalone_validation_resolves_portable_profile_paths(self):
        source = Path("val.py").read_text(encoding="utf-8")
        self.assertIn("resolve_data_options(", source)
        self.assertIn("override_dataset_root(", source)
        for argument in ("--data-root", "--meta", "--support-root"):
            self.assertIn(f"parser.add_argument('{argument}'", source)

    def test_target_count_does_not_depend_on_true_positives(self):
        count_targets = load_val_function("count_targets_per_class")
        stats = [
            [[False] * 10, [False] * 10],
            [0.2, 0.1],
            [1, 4],
            [1, 4, 4],
        ]

        counts = count_targets(stats, nc=5)

        self.assertEqual(counts, [0, 1, 0, 0, 2])

    def test_target_count_is_fixed_length_when_stats_are_empty(self):
        count_targets = load_val_function("count_targets_per_class")

        counts = count_targets([], nc=3)

        self.assertEqual(counts, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
