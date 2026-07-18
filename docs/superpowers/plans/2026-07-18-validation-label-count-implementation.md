# AOFS Validation Label Count Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure AOFS validation reports the real number of ground-truth labels even when the model produces zero True Positives.

**Architecture:** Add one pure counting helper to `val.py`, call it independently of the existing AP gate, and exercise the helper through the existing lightweight AST-based validation contract tests. Keep model inference, loss, AP calculation, data filtering, checkpoint selection, and AMP behavior unchanged.

**Tech Stack:** Python 3.10, NumPy, standard-library `unittest`, Git, GitHub.

---

## File map

- Modify `tests/test_val_contract.py`: execute the pure helper from the real `val.py` AST and cover zero-True-Positive and empty-stats cases without importing CUDA extensions.
- Modify `val.py`: add `count_targets_per_class` and decouple target counting from `stats[0].any()`.
- Modify `MODIFICATIONS_FROM_UPSTREAM.md`: record the plan, production change, tests, RED/GREEN evidence, and final verification.
- No change to `train.py`, dataset files, model definitions, hyperparameters, AMP calls, or server-generated caches and weights.

### Task 1: Reproduce the misleading zero-label summary

**Files:**
- Modify: `tests/test_val_contract.py`
- Test: `tests/test_val_contract.py`

- [ ] **Step 1: Add a lightweight loader for the pure function**

Add NumPy and a helper that executes the exact function definition from `val.py` without importing GPU/native-extension dependencies:

```python
import numpy as np


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
    namespace = {"np": np}
    exec(compile(module, "val.py", "exec"), namespace)
    return namespace[name]
```

- [ ] **Step 2: Add the zero-True-Positive regression test**

Add this method to `ValidationContractTest`:

```python
def test_target_count_does_not_depend_on_true_positives(self):
    count_targets = load_val_function("count_targets_per_class")
    stats = [
        np.zeros((2, 10), dtype=bool),
        np.array([0.2, 0.1]),
        np.array([1, 4]),
        np.array([1, 4, 4]),
    ]

    counts = count_targets(stats, nc=5)

    np.testing.assert_array_equal(counts, np.array([0, 1, 0, 0, 2]))
```

- [ ] **Step 3: Add the empty-stats regression test**

Add this method to `ValidationContractTest`:

```python
def test_target_count_is_fixed_length_when_stats_are_empty(self):
    count_targets = load_val_function("count_targets_per_class")

    counts = count_targets([], nc=3)

    np.testing.assert_array_equal(counts, np.zeros(3, dtype=np.int64))
```

- [ ] **Step 4: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_val_contract -v
```

Expected: the existing three tests pass; both new tests fail with `AssertionError: count_targets_per_class is missing from val.py`. This proves the regression test detects the absent behavior rather than an unrelated import or syntax error.

### Task 2: Implement the minimum production fix

**Files:**
- Modify: `val.py:76-101`
- Modify: `val.py:356-364`
- Test: `tests/test_val_contract.py`

- [ ] **Step 1: Add the pure target-count helper**

Insert after `process_batch` and before `run`:

```python
def count_targets_per_class(stats, nc):
    """Count validation targets independently of prediction correctness."""
    if not len(stats):
        return np.zeros(nc, dtype=np.int64)
    return np.bincount(np.asarray(stats[3], dtype=np.int64), minlength=nc)
```

- [ ] **Step 2: Decouple label counting from the AP condition**

Replace the existing summary block:

```python
stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
else:
    nt = torch.zeros(1)
```

with:

```python
stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
nt = count_targets_per_class(stats, nc)
if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
```

- [ ] **Step 3: Run the focused test and verify GREEN**

Run:

```powershell
python -m unittest tests.test_val_contract -v
```

Expected: five tests pass, zero failures and zero errors.

- [ ] **Step 4: Run syntax verification**

Run:

```powershell
python -m py_compile val.py tests/test_val_contract.py
```

Expected: exit code 0 and no output.

### Task 3: Record every upstream difference and verify the repository

**Files:**
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md`
- Verify: all repository tests

- [ ] **Step 1: Update the modification ledger**

Make these exact ledger updates:

- Mark `docs/superpowers/specs/2026-07-18-validation-label-count-design.md` as user-approved.
- Add `docs/superpowers/plans/2026-07-18-validation-label-count-implementation.md` with its TDD and server-sync scope.
- Extend the `val.py` row to state that ground-truth counting is independent of True Positives.
- Extend the `tests/test_val_contract.py` row to record both new regression cases and the RED/GREEN result.
- Add a dated verification entry containing the exact focused/full test, `py_compile`, and `git diff --check` results.

- [ ] **Step 2: Run the complete unit-test suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: 46 tests pass, zero failures and zero errors.

- [ ] **Step 3: Check whitespace and scope**

Run:

```powershell
git diff --check
git status --short
git diff -- val.py tests/test_val_contract.py MODIFICATIONS_FROM_UPSTREAM.md
```

Expected: `git diff --check` exits 0; status lists only the planned implementation and ledger files; the diff contains no AMP, model, loss, hyperparameter, or dataset changes.

- [ ] **Step 4: Commit the verified implementation**

Run:

```powershell
git add val.py tests/test_val_contract.py MODIFICATIONS_FROM_UPSTREAM.md docs/superpowers/plans/2026-07-18-validation-label-count-implementation.md
git commit -m "Fix validation label counts without true positives"
```

Expected: one implementation commit on `aofs-dual-profile`, followed by a clean `git status --short`.

### Task 4: Publish and validate on the training server

**Files:**
- Publish branch: `aofs-dual-profile`
- Server repository: `/workspace/AOFS_new`
- Server smoke output: `runs/smoke/nwpu_base_cuda_smoke_fixed`

- [ ] **Step 1: Push the local branch**

Run locally:

```powershell
git push origin aofs-dual-profile
```

Expected: GitHub advances `aofs-dual-profile` to the implementation commit.

- [ ] **Step 2: Fast-forward the clean server clone**

Run on the server:

```bash
cd /workspace/AOFS_new
conda activate aofs
git status --short
git pull --ff-only origin aofs-dual-profile
git rev-parse --short HEAD
python -m unittest tests.test_val_contract -v
```

Expected: status is clean before pull, pull fast-forwards, the reported commit matches GitHub, and five focused tests pass.

- [ ] **Step 3: Rerun the one-epoch CUDA smoke under the same paper profile**

Run on an idle GPU:

```bash
export WANDB_MODE=disabled
export WANDB_SILENT=true
export AOFS_DATA_ROOT=/workspace/dataset
export AOFS_BASE_META=/workspace/AOFS_new/data/nwpu_traindict_full.txt
export AOFS_SEED=0

python train.py \
  --cfg1 models/AOFS_s.yaml \
  --cfg2 models/reweight_s.yaml \
  --data data/nwpu_poly.yaml \
  --data-root "$AOFS_DATA_ROOT" \
  --cfgdata cfg/paper/nwpu_base.data \
  --hyp cfg/paper/hyp.finetune_nwpu.yaml \
  --profile paper \
  --stage base \
  --dataset-name nwpu \
  --epochs 1 \
  --batch-size 1 \
  --img 1024 \
  --workers 0 \
  --device 0 \
  --project runs/smoke \
  --name nwpu_base_cuda_smoke_fixed \
  --val-period 1 \
  --checkpoint-metric hbb_fitness
```

Expected: training and validation finish; the validation summary reports a nonzero `Labels` count even if P/R/AP remain zero.

- [ ] **Step 4: Verify smoke artifacts before authorizing full training**

Run:

```bash
find runs/smoke/nwpu_base_cuda_smoke_fixed -maxdepth 2 -type f \
  \( -name "last.pt" -o -name "best.pt" -o -name "run_manifest.json" \) \
  -print
```

Expected: `weights/last.pt`, `weights/best.pt`, and `run_manifest.json` are present. Only after these artifacts and the nonzero label summary are confirmed should the 100-epoch base run begin.
