# NWPU Base Hyperparameter Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, tested NWPU Base hyperparameter profile with momentum `0.937` while preserving paper and robust few-shot momentum `0.999`.

**Architecture:** Keep optimizer selection in stage-specific YAML files rather than adding branching to Python training code. A focused profile test defines the Base/few-shot contract, the reproduction guide points Base training to the new file, and the upstream-difference ledger records both the source changes and the server evidence.

**Tech Stack:** YAML configuration, Python standard-library `unittest`, Markdown documentation, Bash syntax checks, Git.

---

## File map

- Create `cfg/paper/hyp.base_nwpu.yaml`: validated Base-stage optimizer, augmentation, and loss settings.
- Modify `tests/test_profiles.py`: assert Base `0.937` separately from paper/robust few-shot `0.999`.
- Modify `docs/AOFS_REPRODUCTION.md`: make the Base command use the dedicated file and explain the evidence-backed stage split.
- Modify `MODIFICATIONS_FROM_UPSTREAM.md`: record every changed path and the completed server validation.

Scope guard: do not modify Python training/evaluation/model code, Pillow compatibility, few-shot YAML values, weights, datasets, caches, or run outputs in this plan.

### Task 1: Define and test the dedicated Base profile

**Files:**
- Create: `cfg/paper/hyp.base_nwpu.yaml`
- Modify: `tests/test_profiles.py:43-54`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md:86-101,145-151`

- [ ] **Step 1: Add the failing Base-profile test**

Insert this method before `test_profile_hyperparameters_match_the_paper` in `ProfileTest`:

```python
    def test_base_hyperparameters_match_validated_author_code(self):
        values = read_simple_yaml("cfg/paper/hyp.base_nwpu.yaml")
        self.assertEqual(values["lr0"], 0.001)
        self.assertEqual(values["momentum"], 0.937)
        self.assertEqual(values["weight_decay"], 0.0005)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_profiles.ProfileTest.test_base_hyperparameters_match_validated_author_code -v
```

Expected: one error with `FileNotFoundError` for `cfg/paper/hyp.base_nwpu.yaml`.

- [ ] **Step 3: Create the minimal Base YAML**

Create `cfg/paper/hyp.base_nwpu.yaml` with the complete validated settings:

```yaml
# Validated AOFS-S NWPU-R Base-stage settings from the author's public code.
lr0: 0.001
lrf: 0.2
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
theta: 0.5
theta_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.2
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 180.0
translate: 0.1
scale: 0.25
shear: 0.0
perspective: 0.0
flipud: 0.5
fliplr: 0.5
mosaic: 0.75
mixup: 0.1
copy_paste: 0.0
cls_theta: 180
csl_radius: 2.0
```

- [ ] **Step 4: Run the Base and few-shot profile tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_profiles -v
```

Expected: 6 tests pass. The new Base test proves `0.937`, while `test_profile_hyperparameters_match_the_paper` still proves paper and robust few-shot `0.999`.

- [ ] **Step 5: Record Task 1 in the ledger**

Add this row after `cfg/paper/nwpu_base.data`:

```markdown
| 本任务 | `cfg/paper/hyp.base_nwpu.yaml` | 上游不存在；新增经服务器完整对照验证的 NWPU Base 专用配置，使用作者公开代码 momentum `0.937`，其余优化器、增广和损失参数保持不变 | 防止 Base 误用论文文字中的 `0.999` 后长期收敛偏弱；不改变 paper/robust few-shot 配置 | profile 测试先 RED 后 GREEN；完整服务器证据见第 5 节 |
```

Replace the existing `tests/test_profiles.py` row with:

```markdown
| 本任务 | `tests/test_profiles.py` | 上游不存在；新增 profile 身份、500-epoch/禁止提前停止、Base `0.937`、few-shot `0.999`、隔离输出、共享评估及 Bash `{:s}` 模板安全测试；轻量解析避免 PyTorch/PyYAML | 在本机验证服务器配置与 shell 值语义，并防止 Base/few-shot momentum 再次混用 | Base 文件缺失测试先 RED；GREEN 后 profile 共 6 项通过 |
```

- [ ] **Step 6: Check and commit Task 1**

Run:

```bash
git diff --check
git status --short
git add cfg/paper/hyp.base_nwpu.yaml tests/test_profiles.py MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "Add validated NWPU base hyperparameters"
```

Expected: whitespace check exits 0; the commit contains only the Base YAML, profile test, and matching ledger entries.

### Task 2: Make the reproduction guide stage-explicit

**Files:**
- Modify: `docs/AOFS_REPRODUCTION.md:86-120`
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md:99`

- [ ] **Step 1: Replace the Base-stage explanation and command**

Replace the paragraph immediately below `## 3. Base 阶段` with:

```markdown
base 阶段只训练一次，随后所有 shot/seed/profile 使用同一个 base checkpoint。这里按 100 epoch 训练，并用 HBB fitness 选择 base checkpoint；论文对比指标仍在 few-shot 阶段使用 novel OBB mAP@0.5。

论文实验设置写明 SGD momentum 为 `0.999`，但作者公开 NWPU 配置使用 `0.937`。同数据、同模型的服务器完整对照中，Base `0.937` 的独立 HBB mAP@0.5 为 `0.277`，而 Base `0.999` 为 `0.0886`。因此 Base 使用独立的 `cfg/paper/hyp.base_nwpu.yaml`；paper/robust few-shot 仍使用 `0.999`，两阶段不得混用。
```

In the command, replace:

```bash
  --hyp cfg/paper/hyp.finetune_nwpu.yaml \
```

with:

```bash
  --hyp cfg/paper/hyp.base_nwpu.yaml \
```

After the `AOFS_BASE_WEIGHT` example, add:

```markdown
服务器已验收的用户自训练 checkpoint 位于 `runs/base/nwpu_aofs_s_m0937/weights/best.pt`，对应 epoch 69；该二进制是运行生成物，不提交到 Git。新环境应按上面的规范命令重新生成，已有服务器实验可直接把 `AOFS_BASE_WEIGHT` 指向该文件。
```

- [ ] **Step 2: Update the guide row in the ledger**

Replace the `docs/AOFS_REPRODUCTION.md` row with:

```markdown
| 本任务 | `docs/AOFS_REPRODUCTION.md` | 上游不存在；新增同步边界、服务器扩展重建、数据预检、Base `0.937`/few-shot `0.999` 分离、paper/robust 训练、独立 OBB 复评、三 seed 汇总和论文目标 | 给出完整实验的唯一操作路径，并防止已证实偏弱的 Base 参数再次进入九组完整实验 | 已对照当前 CLI/profile/output；Base 独立验证为 HBB mAP@0.5 `0.277`、HBB mAP@0.5:0.95 `0.0979` |
```

- [ ] **Step 3: Verify the documented paths and values**

Run:

```bash
rg -n "hyp\.base_nwpu|hyp\.finetune_nwpu|0\.937|0\.999|0\.277|0\.0886" docs/AOFS_REPRODUCTION.md cfg/paper/hyp.base_nwpu.yaml cfg/paper/hyp.finetune_nwpu.yaml
```

Expected: the Base command references `hyp.base_nwpu.yaml`; the stage-specific momentum values and server evidence are visible.

- [ ] **Step 4: Check and commit Task 2**

Run:

```bash
git diff --check
git add docs/AOFS_REPRODUCTION.md MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "Document stage-specific NWPU momentum"
```

Expected: whitespace check exits 0 and the commit contains only the guide and its ledger update.

### Task 3: Run the complete regression and close the ledger

**Files:**
- Modify: `MODIFICATIONS_FROM_UPSTREAM.md:135-151`

- [ ] **Step 1: Run the focused profile suite again**

Run `python -m unittest tests.test_profiles -v`.

Expected: 6 tests pass, 0 failures, 0 errors.

- [ ] **Step 2: Run the complete unit-test suite**

Run `python -m unittest discover -s tests -v`.

Expected: 47 tests pass, 0 failures, 0 errors.

- [ ] **Step 3: Parse and compare the three YAML files**

Run:

```bash
python - <<'PY'
from pathlib import Path

def read_values(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            values[key] = float(value)
    return values

expected = {
    "cfg/paper/hyp.base_nwpu.yaml": 0.937,
    "cfg/paper/hyp.finetune_nwpu.yaml": 0.999,
    "cfg/robust/hyp.finetune_nwpu.yaml": 0.999,
}

for path, momentum in expected.items():
    values = read_values(path)
    assert values["lr0"] == 0.001, (path, values["lr0"])
    assert values["momentum"] == momentum, (path, values["momentum"])
    assert values["weight_decay"] == 0.0005, (path, values["weight_decay"])
    print(path, values["lr0"], values["momentum"], values["weight_decay"])
PY
```

Expected output:

```text
cfg/paper/hyp.base_nwpu.yaml 0.001 0.937 0.0005
cfg/paper/hyp.finetune_nwpu.yaml 0.001 0.999 0.0005
cfg/robust/hyp.finetune_nwpu.yaml 0.001 0.999 0.0005
```

- [ ] **Step 4: Verify unchanged Bash launchers and repository whitespace**

Run on this Windows workspace:

```powershell
& 'D:\github\Git\bin\bash.exe' -n `
  scripts/train_nwpu_paper.sh `
  scripts/train_nwpu_robust.sh `
  scripts/eval_nwpu_obb.sh
git diff --check HEAD~2
```

Expected: both commands exit 0 with no error output.

- [ ] **Step 5: Append the actual validation evidence to the ledger**

Append these bullets to the current validation section:

```markdown
- Base profile RED：新增测试在 `cfg/paper/hyp.base_nwpu.yaml` 缺失时按预期以 `FileNotFoundError` 失败；新增配置后 `python -m unittest tests.test_profiles -v` 为 6 项通过。
- 完整回归：`python -m unittest discover -s tests -v` 为 47 项通过，0 failure、0 error；三份 YAML 显式解析为 Base `0.937`、paper few-shot `0.999`、robust few-shot `0.999`。
- 服务器受控实验：当前代码、相同数据和 seed 下，100-epoch Base `0.999` 独立 HBB mAP@0.5/0.5:0.95 为 `0.0886/0.0322`；Base `0.937` 的 epoch-69 `best.pt` 独立复评为 `0.277/0.0979`，产物位于 `runs/base/nwpu_aofs_s_m0937/`。10-epoch 对照中 `0.999` 前期更快，故最终决策依据完整 100-epoch 结果。
- Bash 三脚本语法检查与 `git diff --check` 均退出 0；本批次未修改训练、模型、损失、数据或评估代码，也未提交权重和运行目录。
```

- [ ] **Step 6: Commit the verified ledger evidence**

Run:

```bash
git add MODIFICATIONS_FROM_UPSTREAM.md
git commit -m "Record validated NWPU base profile"
git status --short
```

Expected: commit succeeds and final `git status --short` prints no entries.
