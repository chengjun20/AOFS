# AOFS NWPU-R 复现与稳健训练操作手册

本文档对应同一仓库中的两个实验配置：

- `paper`：AOFS-S 核心模型、论文训练预算和论文超参数；support 实例按 seed 随机抽样，不限制同一图像贡献数。
- `robust`：模型、损失、优化器和 500 epoch 预算与 `paper` 相同；仅把同一类别在单张图像中的 support 实例数默认限制为 1，并使用独立输出目录。

论文数值能否达到必须由目标服务器上的完整实验决定。代码修复消除了已知的训练长度、路径、验证、类别映射和统计口径错误，但在完成真实 GPU 三 seed 训练前不能保证达到论文表格数值。

## 1. 同步范围与环境安装

在服务器使用本仓库源码。同步时包含 `aofs/`、`cfg/`、`data/`、`DOTA_devkit/`、`models/`、`scripts/`、`tools/`、`utils/`、`train.py`、`val.py` 和依赖文件；不要从本机复制以下机器相关或可再生成内容：

- `__pycache__/`、`*.pyc`、`build/`、`*.egg-info/`；
- 本机编译的 `*.so`；
- `runs/`、`saved_weights/`、`weights/`，除非明确需要传输某个 base checkpoint。

安装 Python/PyTorch 依赖后，在服务器上按服务器自己的 PyTorch/CUDA 版本重建扩展：

```bash
pip install -r requirements.txt

cd utils/nms_rotated
python setup.py build_ext --inplace
cd ../..

cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
cd ..
```

先确认扩展能导入：

```bash
python -c "from utils.nms_rotated import nms_rotated_wrapper; print('rotated NMS OK')"
python -c "from DOTA_devkit import polyiou; print('polygon IoU OK')"
```

## 2. 数据目录预检

设置数据根目录和 OBB 评估文件：

```bash
export AOFS_DATA_ROOT=/absolute/path/to/NWPU-R
export AOFS_OBB_ANNOPATH="$AOFS_DATA_ROOT/evaluation/labelTxt/{:s}.txt"
export AOFS_OBB_IMAGESET="$AOFS_DATA_ROOT/imgnamefile.txt"
```

期望至少存在：

```text
$AOFS_DATA_ROOT/
├── training.txt
├── evaluation.txt
├── imgnamefile.txt
├── training/
│   ├── images/
│   └── labelTxt/
└── evaluation/
    ├── images/
    └── labelTxt/
```

`training.txt` 和 `evaluation.txt` 每行是图像路径；`labelTxt/*.txt` 使用 DOTA polygon 格式：8 个坐标、类别名、difficult。`imgnamefile.txt` 每行只能是 evaluation 图像 ID，不是完整路径。执行预检：

```bash
test -f "$AOFS_DATA_ROOT/training.txt"
test -f "$AOFS_DATA_ROOT/evaluation.txt"
test -f "$AOFS_OBB_IMAGESET"
test -d "$AOFS_DATA_ROOT/training/images"
test -d "$AOFS_DATA_ROOT/training/labelTxt"
test -d "$AOFS_DATA_ROOT/evaluation/images"
test -d "$AOFS_DATA_ROOT/evaluation/labelTxt"
```

如尚未生成图像列表、逐类别标签和 base meta 字典：

```bash
python tools/label_nwpu.py "$AOFS_DATA_ROOT"
python tools/label_1c_nwpu.py "$AOFS_DATA_ROOT"
(cd tools && python gen_dict_file.py "$AOFS_DATA_ROOT" nwpu)
export AOFS_BASE_META="$PWD/data/nwpu_traindict_full.txt"
```

## 3. Base 阶段

base 阶段只训练一次，随后所有 shot/seed/profile 使用同一个 base checkpoint。这里按 100 epoch 训练，并用 HBB fitness 选择 base checkpoint；论文对比指标仍在 few-shot 阶段使用 novel OBB mAP@0.5。

论文实验设置写明 SGD momentum 为 `0.999`，但作者公开 NWPU 配置使用 `0.937`。同数据、同模型的服务器完整对照中，Base `0.937` 的独立 HBB mAP@0.5 为 `0.277`，而 Base `0.999` 为 `0.0886`。因此 Base 使用独立的 `cfg/paper/hyp.base_nwpu.yaml`；paper/robust few-shot 仍使用 `0.999`，两阶段不得混用。

```bash
export AOFS_SEED=0
export AOFS_BASE_META="$PWD/data/nwpu_traindict_full.txt"

python train.py \
  --cfg1 models/AOFS_s.yaml \
  --cfg2 models/reweight_s.yaml \
  --data data/nwpu_poly.yaml \
  --data-root "$AOFS_DATA_ROOT" \
  --cfgdata cfg/paper/nwpu_base.data \
  --hyp cfg/paper/hyp.base_nwpu.yaml \
  --profile paper \
  --stage base \
  --dataset-name nwpu \
  --epochs 100 \
  --batch-size 1 \
  --img 1024 \
  --device 0 \
  --project runs/base \
  --name nwpu_aofs_s \
  --val-period 10 \
  --checkpoint-metric hbb_fitness
```

后续命令中的 base 权重示例为：

```bash
export AOFS_BASE_WEIGHT="$PWD/runs/base/nwpu_aofs_s/weights/best.pt"
test -f "$AOFS_BASE_WEIGHT"
```

服务器已验收的用户自训练 checkpoint 位于 `runs/base/nwpu_aofs_s_m0937/weights/best.pt`，对应 epoch 69；该二进制是运行生成物，不提交到 Git。新环境应按上面的规范命令重新生成，已有服务器实验可直接把 `AOFS_BASE_WEIGHT` 指向该文件。

## 4. Paper 与 robust 三 seed 训练

每个 tuning 配置使用 `max_epoch=50000, repeat=100`，即 500 个训练 epoch。默认每 10 epoch 做一次 OBB 验证，最终 epoch 无条件验证。`best.pt` 由 NWPU novel 三类 OBB mAP@0.5 选择。

先对 3/5/10-shot、seed 0/1/2 启动 `paper`：

```bash
for shot in 3 5 10; do
  for seed in 0 1 2; do
    bash scripts/train_nwpu_paper.sh "$shot" "$seed" 0 "$AOFS_BASE_WEIGHT"
  done
done
```

再运行同训练预算的 `robust`：

```bash
for shot in 3 5 10; do
  for seed in 0 1 2; do
    bash scripts/train_nwpu_robust.sh "$shot" "$seed" 0 "$AOFS_BASE_WEIGHT"
  done
done
```

主要训练输出为：

```text
runs/paper/nwpu_<shot>shot_seed_<seed>/
runs/robust/nwpu_<shot>shot_seed_<seed>/
├── run_manifest.json
├── best_obb_metrics.json
└── weights/
    ├── best.pt
    └── last.pt

runs/splits/<profile>/<shot>shot/seed_<seed>/
├── audit.json
├── meta.txt
└── support_labels/
```

如果同名目录已存在，训练入口会自动创建递增目录，避免覆盖。中断后只能对同一 profile/dataset/shot/seed/stage 使用 `--resume <last.pt>`；把 base 权重作为 few-shot 初始权重时不要使用 `--resume`。

## 5. 独立复评

训练完成后，可对任意 checkpoint 重新生成确定性 support split、导出 polygon JSON 并计算 OBB AP：

```bash
bash scripts/eval_nwpu_obb.sh \
  paper 5 0 0 \
  runs/paper/nwpu_5shot_seed_0/weights/best.pt
```

默认结果位于 `runs/eval/nwpu_paper_5shot_seed_0/`。该脚本调用 `val.py --save-json` 后，再调用共享的 `DOTA_devkit/dota_evaluation_task1.py --dataset nwpu`，不会沿用作者脚本中错误的 DIOR 类别表，也不会从 mAP 分母中删除无预测类别。

## 6. 三 seed 汇总

每个训练 run 根目录的 `best_obb_metrics.json` 与其 `run_manifest.json` 配对。下面示例汇总 paper 5-shot：

```bash
python tools/summarize_obb_runs.py \
  runs/paper/nwpu_5shot_seed_0/best_obb_metrics.json \
  runs/paper/nwpu_5shot_seed_1/best_obb_metrics.json \
  runs/paper/nwpu_5shot_seed_2/best_obb_metrics.json \
  --output runs/paper/nwpu_5shot_summary.json
```

工具使用样本标准差，并拒绝重复 seed、少于三个 seed、混合 profile、混合 shot、混合 dataset 或混合 stage。

## 7. 指标解释与论文目标

训练日志中的 `HBBmAP@.5` 是水平外接框指标；历史结果还常常是所有类别平均。它们都不能直接与论文的 novel OBB mAP@0.5 比较。有效对比值是 `best_obb_metrics.json` 中的 `novel_map50`，其 novel 类固定为 `airplane`、`baseball-diamond` 和 `tennis-court`，无预测类别按 AP=0 计入固定分母。报告还包含逐类 `prediction_counts` 和排除 difficult 标注后的 `positive_counts`，用于检查异常空类或数据版本差异。

论文中 AOFS-S 在 NWPU-R 上报告的 novel OBB mAP@0.5（百分数）为：

| Shot | 论文 AOFS-S |
|---:|---:|
| 3 | 58.6 ± 2.0 |
| 5 | 64.3 ± 0.7 |
| 10 | 76.1 ± 3.9 |

仓库现在具备与该口径对齐的训练预算、超参数、三 seed 隔离和评估流程，但最终是否达到这些数值仍取决于数据版本/切分、base checkpoint、GPU 数值环境以及完整训练结果，应以服务器实测的 mean ± sample std 为准。

## 8. 建议的服务器验收顺序

1. 重建并导入 polygon IoU 与 rotated NMS 扩展。
2. 运行所有 `--help` 和 `python -m unittest discover -s tests -v`。
3. 只生成一个 3-shot seed，检查 `audit.json`、`meta.txt` 和 support 标签。
4. 用临时缩短配置运行一次 CUDA smoke training，确认 forward/backward、验证 JSON、`obb_metrics.json` 和 checkpoint 均生成。
5. 完成 paper 的 3/5/10-shot × seeds 0/1/2，再汇总论文口径。
6. 在相同 base 权重和预算下运行 robust，单独报告，不与 paper 结果混算。
