# AOFS Strict Numerical Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the AOFS-S NWPU-R Table III 3/5/10-shot numbers under an auditable protocol, while separating the author's public-code behavior from corrected scientific evaluation.

**Architecture:** Keep the current `aofs-dual-profile` experiments unchanged and create an isolated worktree at the author's exact public commit `289af16d3dbbdfd1c0f98fb0dddab350be77493a`. Treat missing author assets as a hard evidence gate. Run the public implementation and the corrected evaluator as separately named tracks; never select a checkpoint on the final evaluation set unless the author confirms that protocol.

**Tech Stack:** Git worktrees, Conda, Python 3.8, PyTorch 1.10.1, CUDA 11.3, AOFS-S, NWPU-R, rotated NMS, polygon IoU, VOC 2007 AP@0.5.

---

## Reproduction claim boundary

The following claim levels must not be mixed:

1. `author-public-raw`: the behavior of the files committed by the author.
2. `author-public-runnable`: the same behavior plus only environment/runtime compatibility patches required to execute it.
3. `paper-protocol`: the paper text plus author-supplied split, support, checkpoint, and evaluation artifacts.
4. `corrected-protocol`: the current fixed NWPU evaluator and fully recorded three-seed protocol.

Only level 3 may be called a strict numerical reproduction of Table III. If the author does not provide the missing artifacts, the final conclusion must be “public-release reconstruction,” not “strict paper reproduction.”

Known public conflicts that require separate experiment identities:

| Item | Paper | README | Tracked public file |
|---|---|---|---|
| SGD momentum | `0.999` | points to public YAML | `0.937` |
| Few-shot physical epochs | not stated | `10000/100 = 100` | `50000/100 = 500` |
| 3/5-shot config | results reported | described by editing 10-shot | not committed |
| Repeated runs | three runs, mean ± std | no seeds supplied | generator fixes seed `2018` |
| Checkpoint selection | not stated | evaluates a named tuning weight | `--noval` validates only the final epoch |
| NWPU OBB evaluator | novel OBB mAP@0.5 | invokes public script | public entry uses a DIOR class table and variable denominator |

### Task 1: Freeze the completed exploratory run

**Files:**
- Create on server: `/workspace/reproduction/current_3shot_seed0.sha256`
- Preserve: `/workspace/AOFS_new/runs/paper/nwpu_3shot_seed_0/`

- [ ] **Step 1: Record source and checkpoint hashes**

```bash
cd /workspace/AOFS_new
mkdir -p /workspace/reproduction
{
  git rev-parse HEAD
  git status --porcelain
  sha256sum \
    runs/base/nwpu_aofs_s_m0937/weights/best.pt \
    runs/paper/nwpu_3shot_seed_0/weights/best.pt \
    runs/paper/nwpu_3shot_seed_0/weights/last.pt \
    runs/paper/nwpu_3shot_seed_0/run_manifest.json \
    runs/paper/nwpu_3shot_seed_0/best_obb_metrics.json
} > /workspace/reproduction/current_3shot_seed0.sha256
```

Expected: the file records commit `379906c...`, a clean/dirty state, and five SHA-256 lines.

- [ ] **Step 2: Label this run correctly**

Record it as:

```text
corrected-protocol / exploratory / seed0 / evaluation-selected-best
```

Do not use its `77.694%` value as an author-paper reproduction result.

### Task 2: Request the missing author evidence

**Files:**
- Create: `docs/reproduction/AUTHOR_ARTIFACT_REQUEST.md`
- Create after receipt: `reproduction/author_artifacts.sha256`
- Create after receipt: `reproduction/author_protocol.json`

- [ ] **Step 1: Request the exact Table III artifacts**

Ask the authors for all of the following:

```text
1. Exact NWPU-R archive/version and train/evaluation image-ID lists.
2. Cropping parameters and the exact annotation files used for Table III.
3. AOFS-S Base checkpoint used to initialize Table III fine-tuning.
4. 3/5/10-shot meta files and per-class support labels for each of the three runs.
5. Random seeds and whether the three runs resampled support or only changed training randomness.
6. Complete Base and few-shot config files, including momentum, max_epoch, repeat and batch size.
7. Checkpoint rule: final epoch, best validation epoch, or another rule.
8. Whether a validation set separate from the reported evaluation set existed.
9. Exact OBB evaluator, class order, AP interpolation, confidence threshold and NMS threshold.
10. Per-run and per-class Table III AP values before rounding.
```

- [ ] **Step 2: Apply the hard evidence gate**

Proceed with a strict `paper-protocol` claim only if items 1–9 are supplied. If any are unavailable, proceed with `author-public-runnable` and state the missing items explicitly.

- [ ] **Step 3: Hash every supplied artifact**

```bash
find /workspace/author_artifacts -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > /workspace/AOFS_author_289af16/reproduction/author_artifacts.sha256
```

Expected: one stable hash line per supplied file.

### Task 3: Create an immutable author-code worktree

**Files:**
- Create on server: `/workspace/AOFS_author_289af16/`
- Do not modify: `/workspace/AOFS_new/`

- [ ] **Step 1: Add the exact upstream worktree**

```bash
cd /workspace/AOFS_new
git worktree add \
  /workspace/AOFS_author_289af16 \
  289af16d3dbbdfd1c0f98fb0dddab350be77493a
cd /workspace/AOFS_author_289af16
git rev-parse HEAD
git status --porcelain
```

Expected:

```text
289af16d3dbbdfd1c0f98fb0dddab350be77493a
```

and no status output.

- [ ] **Step 2: Create the compatibility branch only if the exact environment cannot execute upstream**

```bash
git switch -c baseline/author-public-runnable
```

Permitted patches are limited to Python/NumPy/PyTorch API compatibility, extension imports, and restoring the author's already-present validation-loader construction. Do not add the new support generator, periodic evaluation, OBB checkpoint selection, profile logic, or optimizer-state isolation.

- [ ] **Step 3: Commit each compatibility patch separately**

```bash
git add \
  train.py dataset.py image.py models/common.py \
  utils/datasets.py utils/general.py utils/loss.py \
  utils/nms_rotated \
  DOTA_devkit/polyiou.py DOTA_devkit/polyiou_wrap.cxx
git commit -m "compat: run AOFS upstream on the reproduction server"
```

The commit message and ledger must name the exact runtime error fixed and state that model mathematics are unchanged.

### Task 4: Recreate the author environment

**Files:**
- Create on server: `/workspace/AOFS_author_289af16/reproduction/conda-explicit.txt`
- Create on server: `/workspace/AOFS_author_289af16/reproduction/runtime.txt`

- [ ] **Step 1: Create the README environment**

```bash
conda create -n aofs_author python=3.8 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate aofs_author
conda install \
  pytorch==1.10.1 \
  torchvision==0.11.2 \
  torchaudio==0.10.1 \
  cudatoolkit=11.3 \
  -c pytorch -c conda-forge -y
python -m pip install numpy==1.23.5
python -m pip install -r requirements.txt
```

- [ ] **Step 2: Rebuild native extensions**

```bash
cd /workspace/AOFS_author_289af16/utils/nms_rotated
python setup.py develop
cd /workspace/AOFS_author_289af16/DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

- [ ] **Step 3: Verify and record the runtime**

```bash
cd /workspace/AOFS_author_289af16
mkdir -p reproduction
python - <<'PY'
import sys
import torch
import torchvision
import nms_rotated_ext
from DOTA_devkit import polyiou

print("python", sys.version)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("nms", nms_rotated_ext.__file__)
print("polyiou", polyiou.__file__)
PY
conda list --explicit > reproduction/conda-explicit.txt
nvidia-smi > reproduction/runtime.txt
nvcc --version >> reproduction/runtime.txt
```

Expected: PyTorch `1.10.1`, torchvision `0.11.2`, CUDA runtime `11.3`, CUDA available, and both extensions import successfully.

### Task 5: Verify the exact NWPU-R release and split

**Files:**
- Use: `/workspace/datasets/NWPU-R_author/`
- Create: `reproduction/nwpu-r.sha256`
- Create: `reproduction/nwpu-r-audit.json`

- [ ] **Step 1: Use a fresh official archive**

Do not silently substitute `/workspace/dataset`. Extract the author-supplied or official linked archive into `/workspace/datasets/NWPU-R_author`.

- [ ] **Step 2: Hash the complete dataset**

```bash
find /workspace/datasets/NWPU-R_author -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > reproduction/nwpu-r.sha256
```

- [ ] **Step 3: Audit image/label pairing and leakage**

The audit must fail when any of these conditions holds:

```text
training list image missing
evaluation list image missing
image without matching label
label without matching image
exact file stem shared by train and evaluation
original scene identifier shared across splits
identical image hash shared across splits
class outside the ten NWPU classes
annotation row not containing 8 coordinates, class and difficult flag
```

- [ ] **Step 4: Compare author-supplied IDs**

Require exact set equality between the supplied Table III train/evaluation IDs and the audited local IDs. A merely equal file count is not sufficient.

### Task 6: Reconstruct only the author's support algorithm

**Files:**
- Modify in author worktree: `tools/gen_fewlist_nwpu.py`
- Create: `cfg/fewtunev5_nwpu_3shot.data`
- Create: `cfg/fewtunev5_nwpu_5shot.data`
- Create: `reproduction/support/seed2018/support.sha256`

- [ ] **Step 1: Make only the README-directed generator edits**

For a public-code reconstruction, change:

```python
few_nums = [3, 5, 10]
DROOT = "/workspace/datasets/NWPU-R_author"
```

Keep `get_bbox_fewlist()`, global `random.seed(2018)`, image removal, and all-ten-class counting unchanged.

- [ ] **Step 2: Generate and hash support data**

```bash
cd /workspace/AOFS_author_289af16
mkdir -p reproduction/support/seed2018
python tools/gen_fewlist_nwpu.py --type box
find /workspace/datasets/NWPU-R_author/nwpulist \
     /workspace/datasets/NWPU-R_author/labels_1c \
     -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > reproduction/support/seed2018/support.sha256
```

- [ ] **Step 3: Prefer author-supplied support files for `paper-protocol`**

If the hashes differ from author-supplied Table III files, keep both sets and name the generated set `author-public-seed2018`; never relabel it as the paper split.

### Task 7: Resolve public training-budget and momentum conflicts with named runs

**Files:**
- Create: `cfg/reproduction/author_config_500_3shot.data`
- Create: `cfg/reproduction/author_config_500_5shot.data`
- Create: `cfg/reproduction/author_config_500_10shot.data`
- Create: `cfg/reproduction/author_readme_100_3shot.data`
- Create: `cfg/reproduction/author_readme_100_5shot.data`
- Create: `cfg/reproduction/author_readme_100_10shot.data`
- Preserve: `data/hyps/hyp.finetune_nwpu.yaml`

- [ ] **Step 1: Define the tracked-config run**

Use:

```text
momentum=0.937
Base epochs=100
few-shot max_epoch=50000
repeat=100
--noval
```

Name it `author-config-500`.

- [ ] **Step 2: Define the README run**

Use:

```text
momentum=0.937
Base epochs=100
few-shot max_epoch=10000
repeat=100
--noval
```

Name it `author-readme-100`.

- [ ] **Step 3: Define the paper run only from author clarification**

Use `momentum=0.999` and the author-confirmed schedule, support files, Base checkpoint and checkpoint rule. Do not infer missing values from whichever public source gives the preferred result.

### Task 8: Run the minimum experiment matrix

**Files:**
- Create on server: `runs/author_public/`
- Create on server after author evidence: `runs/paper_protocol/`

- [ ] **Step 1: Run one Base model per protocol**

For `author-public`, use the public AOFS-S and reweight-S YAMLs, public hyperparameters, batch size 1, image size 1024, and final-only validation.

- [ ] **Step 2: Run the ambiguity pilot before nine full jobs**

Run only:

```text
author-config-500 / 3-shot / seed2018
author-readme-100 / 3-shot / seed2018
```

Evaluate their final checkpoints. This determines the practical impact of the README/config conflict without spending nine full runs first.

- [ ] **Step 3: Run the strict Table III matrix after the evidence gate**

```text
3-shot × 3 author-supplied runs
5-shot × 3 author-supplied runs
10-shot × 3 author-supplied runs
```

This is nine few-shot jobs plus the author-confirmed Base job or supplied Base checkpoint.

- [ ] **Step 4: Never choose the maximum over repeated final-evaluation measurements**

Use the author-confirmed checkpoint rule. If no separate validation set is supplied, evaluate the fixed final checkpoint exactly once for the primary result. Periodic evaluation may be retained only as a diagnostic and may not choose the reported model.

### Task 9: Evaluate every prediction through two explicitly named paths

**Files:**
- Preserve: author raw evaluator output
- Create: corrected NWPU evaluator output

- [ ] **Step 1: Run `author_raw_metric`**

Run the author's evaluator unchanged and preserve its output as evidence of public-code behavior.

- [ ] **Step 2: Run `nwpu_corrected_metric` on the identical prediction JSON**

Use the fixed NWPU ten-class order, novel classes:

```text
airplane
baseball-diamond
tennis-court
```

and include missing-prediction classes as AP `0` in a fixed three-class denominator.

- [ ] **Step 3: Report the evaluator disagreement**

The paper comparison must state which evaluator reproduces the author's per-run raw numbers. If the author cannot supply the evaluator or raw per-run numbers, report both values and mark the paper metric implementation as unverified.

### Task 10: Apply numerical-reproduction acceptance criteria

**Files:**
- Create: `reproduction/table3_aofs_s.json`
- Create: `reproduction/STRICT_REPRODUCTION_REPORT.md`
- Update: `MODIFICATIONS_FROM_UPSTREAM.md`

- [ ] **Step 1: Record primary targets**

```text
AOFS-S 3-shot:  58.6 ± 2.0
AOFS-S 5-shot:  64.3 ± 0.7
AOFS-S 10-shot: 76.1 ± 3.9
```

- [ ] **Step 2: Compute three-run sample statistics**

For each shot, store all three raw novel OBB mAP@0.5 values, arithmetic mean, sample standard deviation, per-class AP, checkpoint hash, support hash, dataset hash and source commit.

- [ ] **Step 3: Classify the outcome**

Use these exact labels:

```text
strictly reproduced:
  author artifacts and protocol match, and unrounded per-run values agree
  within 0.1 percentage point or the author-confirmed numerical tolerance

numerically consistent:
  protocol is matched, but only rounded aggregate targets are available;
  the reproduced mean lies inside the paper's reported mean ± standard deviation

public-release reconstruction only:
  one or more author-only artifacts or protocol decisions remain unavailable

not reproduced:
  protocol is matched and the three-run result falls outside the reported
  interval, with evaluator and artifacts independently verified
```

- [ ] **Step 4: Preserve negative and ambiguous results**

Do not discard runs that disagree with the paper. The final report must include the README/config conflict, raw/corrected evaluator difference, all seeds, failed runs, and the reason each result receives its claim label.
