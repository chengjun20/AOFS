import ast
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
