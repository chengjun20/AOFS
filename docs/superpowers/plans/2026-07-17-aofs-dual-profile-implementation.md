# AOFS Dual-Profile Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build paper-compatible and robust AOFS training profiles in one repository, with portable server paths, reproducible few-shot splits, correct NWPU OBB evaluation, metric-based checkpoints, and a complete upstream-difference ledger.

**Architecture:** Put deterministic, CPU-testable experiment logic in a small `aofs` package and keep `train.py`, `val.py`, and legacy DOTA entry points as adapters. Both profiles share the author model and loss implementation; only configuration, split policy, validation cadence, cache/output identity, and launch commands differ.

**Tech Stack:** Python 3.10, standard-library `unittest`, NumPy, PyTorch/YOLOv5 training code, DOTA polygon IoU extension on the training server, Bash launch scripts, YAML and Darknet-style `.data` files.

---

### Task 1: Portable experiment configuration and run identity

**Files:**
- Create: `aofs/__init__.py`
- Create: `aofs/experiment.py`
- Create: `tests/__init__.py`
- Create: `tests/test_experiment.py`
- Modify: `util.py:705`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write failing tests for path expansion, epoch conversion, validation cadence, and resume identity**

```python
import os
import unittest
from unittest.mock import patch

from aofs.experiment import (
    build_experiment_identity,
    compute_tuning_epochs,
    resolve_data_options,
    should_validate,
    validate_resume_identity,
)


class ExperimentTest(unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "AOFS_SPLIT_ROOT"):
            resolve_data_options({"meta": "${AOFS_SPLIT_ROOT}/meta.txt"})

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
```

- [x] **Step 2: Run the tests and verify the missing module failure**

Run: `python -m unittest tests.test_experiment -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'aofs'`.

- [x] **Step 3: Implement the pure experiment helpers**

```python
# aofs/experiment.py
import json
import math
import os
import re
import subprocess
from pathlib import Path

PATH_KEYS = {"train", "valid", "meta", "support_root"}
VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_data_options(options, data_root=None, overrides=None):
    resolved = dict(options)
    if data_root:
        root = Path(data_root).expanduser()
        resolved["train"] = str(root / "training.txt")
        resolved["valid"] = str(root / "evaluation.txt")
    if overrides:
        resolved.update({key: value for key, value in overrides.items() if value not in (None, "")})
    for key in PATH_KEYS & resolved.keys():
        value = os.path.expanduser(os.path.expandvars(str(resolved[key])))
        missing = VARIABLE_PATTERN.findall(value)
        if missing:
            raise ValueError(f"Unresolved environment variable(s) in {key}: {', '.join(missing)}")
        resolved[key] = os.path.normpath(value)
    return resolved


def compute_tuning_epochs(max_epoch, repeat):
    if int(repeat) <= 0:
        raise ValueError("repeat must be greater than zero")
    return int(math.ceil(int(max_epoch) / int(repeat)))


def should_validate(epoch, epochs, noval, period):
    final_epoch = epoch + 1 == epochs
    if final_epoch:
        return True
    if noval:
        return False
    return period > 0 and (epoch + 1) % period == 0


def build_experiment_identity(profile, dataset, shot, seed, stage):
    return {"profile": profile, "dataset": dataset, "shot": int(shot), "seed": int(seed), "stage": stage}


def validate_resume_identity(expected, actual):
    if not actual:
        raise ValueError("Checkpoint has no experiment identity; use it as initial weights, not --resume")
    differences = [key for key, value in expected.items() if actual.get(key) != value]
    if differences:
        raise ValueError("Resume identity mismatch: " + ", ".join(differences))


def write_run_manifest(path, identity, options, data_options):
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    payload = {"identity": identity, "options": vars(options), "data_options": data_options,
               "git_revision": revision, "git_dirty": dirty}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
```

Change `read_data_cfg` to split only on the first `=` so environment-bearing values remain valid:

```python
key, value = line.split('=', 1)
```

- [x] **Step 4: Run the experiment tests and verify they pass**

Run: `python -m unittest tests.test_experiment -v`

Expected: six tests pass.

- [x] **Step 5: Record Task 1 paths in the upstream-difference ledger**

Add one `本任务` row for each of `aofs/__init__.py`, `aofs/experiment.py`, `tests/__init__.py`, `tests/test_experiment.py`, and `util.py`, including the test command above.

- [ ] **Step 6: Commit the self-contained configuration layer**

```bash
git add aofs/__init__.py aofs/experiment.py tests/__init__.py tests/test_experiment.py util.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "feat: add reproducible AOFS experiment configuration"
```

Expected: commit succeeds after repository-local Git identity is configured; if identity remains unset, preserve the verified files without staging unrelated historical changes.

### Task 2: Dataset registry and correct OBB aggregation

**Files:**
- Create: `aofs/datasets.py`
- Create: `aofs/obb_eval.py`
- Create: `tests/test_obb_eval.py`
- Modify: `DOTA_devkit/dota_evaluation_task1.py:281`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write failing tests for NWPU mapping and zero-AP missing classes**

```python
import tempfile
import unittest
from pathlib import Path

from aofs.datasets import get_dataset_spec
from aofs.obb_eval import aggregate_group_aps, evaluate_detection_files


class ObbEvaluationTest(unittest.TestCase):
    def test_nwpu_registry_uses_ten_classes_and_expected_novel_split(self):
        spec = get_dataset_spec("nwpu")
        self.assertEqual(len(spec.classes), 10)
        self.assertEqual(spec.novel, ("airplane", "baseball-diamond", "tennis-court"))
        self.assertEqual(set(spec.classes), set(spec.novel) | set(spec.base))

    def test_missing_detection_class_contributes_zero_to_denominator(self):
        metrics = aggregate_group_aps(
            {"airplane": 0.6, "baseball-diamond": 0.3},
            ("airplane", "baseball-diamond", "tennis-court"),
        )
        self.assertAlmostEqual(metrics, 0.3)

    def test_evaluator_calls_voc_eval_only_for_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Task1_airplane.txt").write_text("img 0.9 0 0 1 0 1 1 0 1\n", encoding="utf-8")
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
            self.assertAlmostEqual(result["novel_map50"], 0.5 / 3)
```

- [x] **Step 2: Run the OBB tests and verify the missing module failure**

Run: `python -m unittest tests.test_obb_eval -v`

Expected: FAIL because `aofs.datasets` and `aofs.obb_eval` do not exist.

- [x] **Step 3: Implement the immutable dataset registry**

```python
# aofs/datasets.py
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    classes: tuple
    novel: tuple
    base: tuple


NWPU_CLASSES = ("airplane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                "basketball-court", "ground-track-field", "harbor", "bridge", "vehicle")
NWPU_NOVEL = ("airplane", "baseball-diamond", "tennis-court")
NWPU = DatasetSpec(NWPU_CLASSES, NWPU_NOVEL,
                   tuple(name for name in NWPU_CLASSES if name not in NWPU_NOVEL))

DIOR_CLASSES = ("airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
                "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", "harbor",
                "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill")
DIOR_NOVEL = ("airplane", "baseballfield", "tenniscourt", "trainstation", "windmill")
DIOR = DatasetSpec(DIOR_CLASSES, DIOR_NOVEL,
                   tuple(name for name in DIOR_CLASSES if name not in DIOR_NOVEL))

DATASETS = {"nwpu": NWPU, "dior": DIOR}


def get_dataset_spec(name):
    try:
        return DATASETS[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {name}") from error
```

- [x] **Step 4: Implement JSON conversion, per-class AP, and novel/base/all metrics**

`aofs/obb_eval.py` must expose `convert_json_predictions`, `aggregate_group_aps`, `evaluate_detection_files`, and `evaluate_obb_predictions`. It must create one empty `Task1_<class>.txt` file per registered class before appending predictions, reject category IDs outside `1..len(classes)`, call VOC 2007 polygon AP only for non-empty files, assign zero otherwise, and write `obb_metrics.json` containing `per_class`, `novel_map50`, `base_map50`, and `all_map50`.

The aggregation function is exactly:

```python
def aggregate_group_aps(per_class, class_names):
    return sum(float(per_class.get(name, 0.0)) for name in class_names) / len(class_names)
```

Replace the hard-coded `main()` selection and shrinking denominator in `DOTA_devkit/dota_evaluation_task1.py` with arguments `--dataset {nwpu,dior}` and a call to `evaluate_obb_predictions`; retain the file's `voc_eval` implementation and pass it into the new adapter.

- [x] **Step 5: Run the OBB tests and CLI help check**

Run: `python -m unittest tests.test_obb_eval -v`

Expected: three tests pass.

Run: `python DOTA_devkit/dota_evaluation_task1.py --help`

Expected: help lists `--dataset`, `--base_path`, `--annopath`, and `--imagesetfile`.

- [ ] **Step 6: Update the ledger and commit the evaluator layer**

```bash
git add aofs/datasets.py aofs/obb_eval.py tests/test_obb_eval.py DOTA_devkit/dota_evaluation_task1.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "fix: evaluate NWPU oriented detections correctly"
```

### Task 3: Reproducible paper and scene-aware robust support splits

**Files:**
- Create: `aofs/fewshot.py`
- Create: `tests/test_fewshot.py`
- Modify: `tools/gen_fewlist_nwpu.py`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write failing tests for exact-shot and per-image limits**

```python
import unittest

from aofs.fewshot import Candidate, select_candidates


class FewShotSelectionTest(unittest.TestCase):
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
```

- [x] **Step 2: Run the test and verify the missing module failure**

Run: `python -m unittest tests.test_fewshot -v`

Expected: FAIL because `aofs.fewshot` does not exist.

- [x] **Step 3: Implement candidate parsing and deterministic selection**

```python
@dataclass(frozen=True)
class Candidate:
    image_path: str
    class_name: str
    polygon: tuple
    difficult: str


def select_candidates(candidates, shot, seed, max_per_image):
    rng = random.Random(seed)
    ordered = list(candidates)
    rng.shuffle(ordered)
    if max_per_image is None:
        if len(ordered) < shot:
            raise ValueError(f"requested {shot} instances but only {len(ordered)} are available")
        return ordered[:shot]
    eligible_images = {item.image_path for item in ordered}
    if len(eligible_images) < math.ceil(shot / max_per_image):
        raise ValueError(f"requested {shot} instances with max_per_image={max_per_image}, "
                         f"but only {len(eligible_images)} eligible images are available")
    counts = collections.Counter()
    selected = []
    for item in ordered:
        if counts[item.image_path] >= max_per_image:
            continue
        selected.append(item)
        counts[item.image_path] += 1
        if len(selected) == shot:
            return selected
    raise ValueError(f"unable to select {shot} instances")
```

Implement `generate_split(dataset_root, output_root, profile, shot, seed, max_per_image)` to parse `training.txt` and `labels/*.txt`, select each NWPU class independently, write `support_labels/<class>/<image-stem>.txt`, `lists/<class>.txt`, `meta.txt`, and `audit.json`. The audit stores the selected instance count, unique image count, maximum instances from one image, source paths, seed, profile, and SHA-256 digest of the manifest.

- [x] **Step 4: Replace the fixed-path legacy generator with a CLI adapter**

`tools/gen_fewlist_nwpu.py` accepts `--data-root`, `--output-root`, `--profile {paper,robust}`, `--shot {3,5,10}`, `--seed`, and `--max-per-image`. `paper` defaults to no cap; `robust` defaults to one instance per source image. It prints the absolute paths to `meta.txt`, `support_labels`, and `audit.json`.

- [x] **Step 5: Verify tests and a synthetic CLI split**

Run: `python -m unittest tests.test_fewshot -v`

Expected: three tests pass.

Run the CLI against a temporary two-class synthetic fixture created by the test helper and verify that rerunning with the same seed produces the same manifest digest.

- [ ] **Step 6: Update the ledger and commit the split generator**

```bash
git add aofs/fewshot.py tests/test_fewshot.py tools/gen_fewlist_nwpu.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "feat: generate reproducible AOFS support splits"
```

### Task 4: Support-label routing and explicit shot/profile configuration

**Files:**
- Create: `tests/test_support_paths.py`
- Modify: `aofs/experiment.py`
- Modify: `cfg.py:90`
- Modify: `dataset.py:483`
- Modify: `train.py:63`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write a failing test for generated support-label paths**

```python
import os
import unittest

from aofs.experiment import support_label_path


class SupportPathTest(unittest.TestCase):
    def test_support_root_separates_seed_specific_labels(self):
        result = support_label_path("/splits/seed7/support_labels", "airplane", "/data/training/images/001.png")
        self.assertEqual(result, os.path.normpath("/splits/seed7/support_labels/airplane/001.txt"))
```

- [x] **Step 2: Run the test and verify `support_label_path` is missing**

Run: `python -m unittest tests.test_support_paths -v`

Expected: FAIL with an import error for `support_label_path`.

- [x] **Step 3: Add the helper and route `MetaDataset` through it**

```python
def support_label_path(support_root, class_name, image_path):
    return os.path.normpath(str(Path(support_root) / class_name / (Path(image_path).stem + ".txt")))
```

In `cfg.__configure_data`, set `cfg.profile`, `cfg.seed`, `cfg.shot`, and `cfg.support_root` from explicit `.data` keys, with legacy filename inference only when `shot` is absent. In `MetaDataset.get_labpath`, use `support_label_path(cfg.support_root, cls_name, imgpath)` when tuning and `support_root` is non-empty; retain the author's path replacement as a legacy fallback.

In `train.py`, add CLI overrides `--profile`, `--stage`, `--dataset-name`, `--shot`, `--seed`, `--data-root`, `--meta`, and `--support-root`; call `resolve_data_options` before `cfg.config_data`.

- [x] **Step 4: Run support and experiment tests**

Run: `python -m unittest tests.test_support_paths tests.test_experiment -v`

Expected: seven tests pass.

- [ ] **Step 5: Update the ledger and commit routing support**

```bash
git add tests/test_support_paths.py aofs/experiment.py cfg.py dataset.py train.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "feat: isolate seed-specific AOFS support labels"
```

### Task 5: Predictable OBB JSON output from validation

**Files:**
- Create: `tests/test_val_contract.py`
- Modify: `val.py:99`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write a failing source-contract test**

```python
import ast
import unittest
from pathlib import Path


class ValidationContractTest(unittest.TestCase):
    def test_run_accepts_prediction_stem_and_writes_even_empty_json(self):
        source = Path("val.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        arguments = [argument.arg for argument in run.args.args]
        self.assertIn("prediction_stem", arguments)
        self.assertIn("if save_json:", source)
        self.assertNotIn("if save_json and len(jdict):", source)
```

- [x] **Step 2: Run the test and verify it fails on the current signature**

Run: `python -m unittest tests.test_val_contract -v`

Expected: FAIL because `prediction_stem` is absent and empty predictions skip JSON creation.

- [x] **Step 3: Make JSON naming explicit and preserve empty prediction files**

Add `prediction_stem=None` to `val.run`. Resolve the name as the explicit stem, otherwise the weight stem, otherwise `predictions`. Change the save guard to `if save_json:` and always dump `jdict`, including `[]`. Run pycocotools only when `is_coco`; NWPU and DIOR OBB scoring is handled by `aofs.obb_eval`.

- [x] **Step 4: Run the validation contract test**

Run: `python -m unittest tests.test_val_contract -v`

Expected: one test passes.

- [ ] **Step 5: Update the ledger and commit the validation output contract**

```bash
git add tests/test_val_contract.py val.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "fix: always emit deterministic OBB prediction JSON"
```

### Task 6: Restore periodic validation and select checkpoints by novel OBB mAP

**Files:**
- Create: `tests/test_train_contract.py`
- Modify: `train.py:250-475`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write failing source-contract tests for validation and checkpoint identity**

```python
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

    def test_validation_loader_is_not_left_commented_out(self):
        self.assertIn("metaset_val = MetaDataset(", self.source)
        self.assertIn("val_loader = create_dataloader(", self.source)
```

- [x] **Step 2: Run the tests and verify all four assertions fail on current code**

Run: `python -m unittest tests.test_train_contract -v`

Expected: four failures describing the missing behavior.

- [x] **Step 3: Add metric and validation CLI parameters**

Add `--val-period` (default 10), `--checkpoint-metric {hbb_fitness,obb_novel_map50}` (default `hbb_fitness` for legacy commands), `--obb-annopath`, and `--obb-imagesetfile`. Reject `obb_novel_map50` unless dataset name, annotation template, and image set are all supplied.

- [x] **Step 4: Restore validation data creation and use `should_validate`**

Create `val_loader` and ensemble `metaset_val` on rank 0. During each epoch:

```python
validate_now = should_validate(epoch, epochs, noval, opt.val_period)
improved = False
if validate_now:
    prediction_stem = f"epoch_{epoch:04d}"
    results, maps, _ = val.run(
        data_dict,
        batch_size=batch_size // WORLD_SIZE * 2,
        imgsz=imgsz,
        model=ema.ema,
        single_cls=single_cls,
        dataloader=val_loader,
        metaset=metaset_val,
        meta_workers=num_workers,
        save_dir=save_dir,
        save_json=opt.checkpoint_metric == "obb_novel_map50",
        prediction_stem=prediction_stem,
        plots=False,
        callbacks=callbacks,
        compute_loss=compute_loss,
    )
    if opt.checkpoint_metric == "obb_novel_map50":
        obb_metrics = evaluate_obb_predictions(
            save_dir / f"{prediction_stem}_obb_predictions.json",
            opt.obb_annopath,
            opt.obb_imagesetfile,
            opt.dataset_name,
            output_dir=save_dir / f"obb_epoch_{epoch:04d}",
        )
        fi = obb_metrics['novel_map50']
    else:
        fi = float(fitness(np.array(results).reshape(1, -1))[0])
    improved = fi > best_fitness
    if improved:
        best_fitness = fi
```

Do not call early stopping on epochs without validation. Always save `last.pt`; save `best.pt` only when `improved`; include `experiment_identity` and the last OBB metrics in the checkpoint. Write `run_manifest.json` before training. On `--resume`, call `validate_resume_identity` before optimizer state is restored.

- [x] **Step 5: Run the train-contract and helper tests**

Run: `python -m unittest tests.test_train_contract tests.test_experiment tests.test_obb_eval -v`

Expected: thirteen tests pass.

- [x] **Step 6: Syntax-compile changed Python entry points**

Run: `python -m py_compile train.py val.py cfg.py dataset.py aofs/experiment.py aofs/datasets.py aofs/obb_eval.py aofs/fewshot.py`

Expected: exit code 0 with no output.

- [ ] **Step 7: Update the ledger and commit training integration**

```bash
git add tests/test_train_contract.py train.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "fix: select AOFS checkpoints with novel OBB validation"
```

### Task 7: Paper and robust profile files plus server launch scripts

**Files:**
- Create: `cfg/paper/nwpu_base.data`
- Create: `cfg/paper/nwpu_3shot.data`
- Create: `cfg/paper/nwpu_5shot.data`
- Create: `cfg/paper/nwpu_10shot.data`
- Create: `cfg/paper/hyp.finetune_nwpu.yaml`
- Create: `cfg/robust/nwpu_3shot.data`
- Create: `cfg/robust/nwpu_5shot.data`
- Create: `cfg/robust/nwpu_10shot.data`
- Create: `cfg/robust/hyp.finetune_nwpu.yaml`
- Create: `scripts/train_nwpu_paper.sh`
- Create: `scripts/train_nwpu_robust.sh`
- Create: `scripts/eval_nwpu_obb.sh`
- Create: `tests/test_profiles.py`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write failing tests that inspect every profile**

```python
import unittest
from pathlib import Path

import yaml

from util import read_data_cfg


class ProfileTest(unittest.TestCase):
    def test_paper_profiles_have_full_training_length_and_explicit_identity(self):
        for shot in (3, 5, 10):
            values = read_data_cfg(Path(f"cfg/paper/nwpu_{shot}shot.data"))
            self.assertEqual(values["profile"], "paper")
            self.assertEqual(int(values["shot"]), shot)
            self.assertEqual(int(values["max_epoch"]), 50000)
            self.assertEqual(int(values["repeat"]), 100)

    def test_paper_hyperparameters_match_the_paper(self):
        values = yaml.safe_load(Path("cfg/paper/hyp.finetune_nwpu.yaml").read_text(encoding="utf-8"))
        self.assertEqual(values["lr0"], 0.001)
        self.assertEqual(values["momentum"], 0.999)
        self.assertEqual(values["weight_decay"], 0.0005)

    def test_profiles_use_environment_paths_and_separate_projects(self):
        paper = Path("scripts/train_nwpu_paper.sh").read_text(encoding="utf-8")
        robust = Path("scripts/train_nwpu_robust.sh").read_text(encoding="utf-8")
        self.assertIn("AOFS_DATA_ROOT", paper)
        self.assertIn("runs/paper", paper)
        self.assertIn("runs/robust", robust)
        self.assertIn("--checkpoint-metric obb_novel_map50", paper)
        self.assertIn("--checkpoint-metric obb_novel_map50", robust)
```

- [x] **Step 2: Run tests and verify profile files are missing**

Run: `python -m unittest tests.test_profiles -v`

Expected: three errors/failures caused by absent profile files.

- [x] **Step 3: Add profile data and hyperparameter files**

Each tuning `.data` file contains explicit `data=nwpu`, `tuning=1`, `profile`, `shot`, `seed=${AOFS_SEED}`, `max_epoch=50000`, `repeat=100`, `train=${AOFS_DATA_ROOT}/training.txt`, `valid=${AOFS_DATA_ROOT}/evaluation.txt`, `meta=${AOFS_SPLIT_ROOT}/meta.txt`, `support_root=${AOFS_SPLIT_ROOT}/support_labels`, and the existing NWPU novel split. Paper and robust hyperparameters initially differ only by filename and comments; both preserve the paper's `lr0=0.001`, `momentum=0.999`, and `weight_decay=0.0005`, so robust algorithm changes remain attributable to sampling and validation.

- [x] **Step 4: Add strict server launch scripts**

Every script starts with `set -euo pipefail`, requires `AOFS_DATA_ROOT`, accepts shot/seed/device/base-weight positional arguments, creates the profile split with `tools/gen_fewlist_nwpu.py`, exports `AOFS_SPLIT_ROOT` and `AOFS_SEED`, and launches `train.py` with isolated `--project`, `--name`, `--dataset-name nwpu`, OBB annotation paths, `--val-period`, and `--checkpoint-metric obb_novel_map50`.

`eval_nwpu_obb.sh` runs `val.py --save-json` with a deterministic prediction stem, then runs `DOTA_devkit/dota_evaluation_task1.py --dataset nwpu` against the generated JSON.

- [x] **Step 5: Run profile tests and Bash syntax checks**

Run: `python -m unittest tests.test_profiles -v`

Expected: three tests pass.

Run on the Linux server or Git Bash: `bash -n scripts/train_nwpu_paper.sh scripts/train_nwpu_robust.sh scripts/eval_nwpu_obb.sh`

Expected: exit code 0 with no output.

- [ ] **Step 6: Update the ledger and commit profile launchers**

```bash
git add cfg/paper cfg/robust scripts tests/test_profiles.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "feat: add paper and robust AOFS server profiles"
```

### Task 8: Multi-seed result summary and operator guide

**Files:**
- Create: `aofs/results.py`
- Create: `tools/summarize_obb_runs.py`
- Create: `tests/test_results.py`
- Create: `docs/AOFS_REPRODUCTION.md`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Write a failing test for mean and sample standard deviation**

```python
import unittest

from aofs.results import summarize_runs


class ResultSummaryTest(unittest.TestCase):
    def test_three_seed_summary_reports_mean_and_sample_std(self):
        summary = summarize_runs([
            {"seed": 0, "novel_map50": 0.50},
            {"seed": 1, "novel_map50": 0.60},
            {"seed": 2, "novel_map50": 0.70},
        ])
        self.assertAlmostEqual(summary["novel_map50_mean"], 0.60)
        self.assertAlmostEqual(summary["novel_map50_std"], 0.10)
        self.assertEqual(summary["seeds"], [0, 1, 2])
```

- [x] **Step 2: Run the result test and verify the missing module failure**

Run: `python -m unittest tests.test_results -v`

Expected: FAIL because `aofs.results` does not exist.

- [x] **Step 3: Implement strict result loading and summary**

```python
import statistics


def summarize_runs(runs):
    if len(runs) < 3:
        raise ValueError("at least three seed results are required")
    seeds = [int(run["seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed values must be unique")
    values = [float(run["novel_map50"]) for run in runs]
    return {"seeds": seeds, "novel_map50_mean": statistics.mean(values),
            "novel_map50_std": statistics.stdev(values), "runs": runs}
```

The CLI accepts three or more `obb_metrics.json` paths plus `--output`, infers seed/profile/shot from each adjacent `run_manifest.json`, rejects mixed profile or shot, and writes JSON with the mean and sample standard deviation.

- [x] **Step 4: Write the server execution guide**

Document repository sync boundaries, required environment variables, extension rebuild commands, data preflight, base training, paper/robust commands for seeds 0/1/2, standalone evaluation, summary generation, expected paper targets, and the distinction between HBB all-class metrics and OBB novel metrics. State explicitly that reaching the paper number is an empirical outcome, not guaranteed before server experiments finish.

- [x] **Step 5: Run result tests and CLI help**

Run: `python -m unittest tests.test_results -v`

Expected: one test passes.

Run: `python tools/summarize_obb_runs.py --help`

Expected: help lists metric paths and `--output`.

- [ ] **Step 6: Update the ledger and commit result tooling and guide**

```bash
git add aofs/results.py tools/summarize_obb_runs.py tests/test_results.py docs/AOFS_REPRODUCTION.md MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "docs: add AOFS server reproduction workflow"
```

### Task 9: Full local verification and change-ledger audit

**Files:**
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`

- [x] **Step 1: Run the complete CPU test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass with zero failures and zero errors.

- [x] **Step 2: Compile every changed Python file**

Run: `python -m py_compile aofs/__init__.py aofs/experiment.py aofs/datasets.py aofs/obb_eval.py aofs/fewshot.py aofs/results.py tools/gen_fewlist_nwpu.py tools/summarize_obb_runs.py DOTA_devkit/dota_evaluation_task1.py cfg.py dataset.py train.py val.py util.py`

Expected: exit code 0 with no output.

- [x] **Step 3: Check profile and evaluation command help**

Run: `python tools/gen_fewlist_nwpu.py --help`

Run: `python tools/summarize_obb_runs.py --help`

Run: `python DOTA_devkit/dota_evaluation_task1.py --help`

Expected: all commands exit 0 and show the arguments specified in earlier tasks.

- [x] **Step 4: Audit every upstream difference**

Run:

```powershell
git status --short
git diff --name-status 289af16d3dbbdfd1c0f98fb0dddab350be77493a
git ls-files --others --exclude-standard
```

Compare every source/config/script/test/document path with Sections 2–4 of `MODIFICATIONS_FROM_UPSTREAM.md`. Add missing paths and update every `本任务` verification field with the exact successful command. Keep build directories, caches, weights, and runs only in the generated-artifact section.

- [x] **Step 5: Review diff boundaries**

Run: `git diff --check`

Expected: no whitespace errors introduced by this task. Review `git diff -- train.py val.py dataset.py cfg.py util.py DOTA_devkit/dota_evaluation_task1.py` to confirm no historical user modification was unintentionally removed.

- [ ] **Step 6: Create the final implementation commit if Git identity is available**

```bash
git add aofs cfg/paper cfg/robust scripts tests tools/gen_fewlist_nwpu.py tools/summarize_obb_runs.py docs/AOFS_REPRODUCTION.md MODIFICATIONS_FROM_UPSTREAM.md train.py val.py dataset.py cfg.py util.py DOTA_devkit/dota_evaluation_task1.py
git commit -m "feat: implement reproducible AOFS paper and robust profiles"
```

Expected: only task-owned files and intentional modifications are included; pre-existing unrelated generated artifacts remain unstaged.

- [x] **Step 7: Record server-only verification still required**

The final handoff lists these unverified local limitations without claiming success: rebuild `DOTA_devkit` polygon IoU and rotated NMS on the server, run one-batch CUDA smoke training, run a short validation that emits OBB JSON/metrics, then run full 3/5/10-shot seeds 0/1/2 and compare the paper profile's novel OBB mAP@0.5 mean/std with the published table.
