# AOFS 阶段工作、实验成果与后续用途记录

记录日期：2026-07-26

上游作者仓库：<https://github.com/lalalayong/AOFS>

作者源码比较基线：`289af16d3dbbdfd1c0f98fb0dddab350be77493a`

当前项目仓库：<https://github.com/chengjun20/AOFS>

当前开发分支：`aofs-dual-profile`

本地记录时提交：`379906c`

服务器工作目录：`/workspace/AOFS_new`

服务器运行环境：Conda `aofs`、Python 3.10.20、PyTorch 2.4.0+cu121、CUDA 12.1、RTX 3090 24 GB（共享）

## 1. 当前项目定位

当前项目不再以“严格复现作者论文表格中的数值”为唯一目标，而是分为两个层次：

1. 对作者公开代码进行系统排查，修正影响运行、训练、验证和评估可信度的问题，建立可审计、可重复运行的 AOFS 基线。
2. 在修正版基线之上设计自己的方法改进，最终形成遥感少样本旋转目标检测的方法型论文。

当前代码应称为：

> 基于作者公开 AOFS 代码构建的修正性复现与可复现实验基线。

在作者没有提供精确数据归档、训练/测试图像 ID、三组 support 样本、Base checkpoint、随机种子和原始评估器之前，当前项目不能称为“严格数值复现了 AOFS 论文表格”。

当前改动均通过 `MODIFICATIONS_FROM_UPSTREAM.md` 记录。后续自己的方法应继续在独立配置、独立实验名称和独立消融中实现，不覆盖作者公开实现或当前修正版基线。

## 2. 最近完成的主要工作

### 2.1 阅读论文并审计作者公开代码

已对论文、README、作者公开配置、训练入口、support 生成脚本、验证逻辑和 OBB 评估脚本进行交叉核对，发现以下主要不一致或风险：

1. 论文文字报告 SGD momentum 为 `0.999`，作者公开 NWPU 配置实际为 `0.937`。
2. README 示例对应约 100 个物理 epoch，而仓库中被跟踪的 few-shot 配置对应约 500 个物理 epoch。
3. 作者没有公开 Table III 使用的精确数据版本、图像划分、三次实验的 support 文件、随机种子、Base 权重和 checkpoint 选择规则。
4. 原验证流程使用 `--noval` 时缺少训练过程中的周期评估；历史本地修改又曾误删最终验证，导致 `best.pt` 缺少可靠指标依据。
5. 作者公开 OBB 评估入口混用了 DIOR 与 NWPU 类别表，并会跳过无预测类别，使 mAP 分母缩小。
6. 原 support 生成和配置大量依赖固定路径、固定 seed 和共享输出目录，不利于不同实验之间隔离。
7. 新版 Python、NumPy、PyTorch 和 CUDA 环境与作者旧代码存在 API、原生扩展和导入兼容问题。

这些问题说明此前效果较差并不完全等同于 AOFS 网络本身无效，其中包含训练预算、Base 权重质量、验证缺失、指标实现和环境兼容等多方面影响。

### 2.2 建立 paper/robust 双配置

新增了两个相互隔离的实验身份：

- `paper`：尽量对齐论文公开描述；few-shot momentum 使用论文文字中的 `0.999`，support 抽样保留 exact-K 语义。
- `robust`：训练预算和超参数与 `paper` 相同，只改变 support 的场景多样性约束，便于后续做单变量对照。

两套配置均显式记录 dataset、shot、seed、stage、support_root、训练预算和输出目录，避免不同 seed 或不同方法覆盖同一批 support 文件和训练结果。

### 2.3 修复训练与断点恢复流程

已经完成的主要训练流程修复包括：

- 支持 `profile/stage/dataset/shot/seed` 显式实验身份。
- 通过环境变量覆盖服务器数据根目录，移除关键训练配置中的机器固定路径。
- 将 3/5/10-shot 训练预算恢复为 `max_epoch=50000, repeat=100`，即 500 个物理 epoch。
- 恢复周期验证和最终 epoch 验证。
- 支持按 novel OBB mAP@0.5 或 HBB fitness 选择 checkpoint。
- 在 checkpoint 和 `run_manifest.json` 中写入实验身份。
- `--resume` 时检查 profile、dataset、shot、seed 和 stage 是否一致。
- 从 Base 进入 few-shot 新阶段时不错误继承上一阶段 optimizer 状态。
- 修复零 True Positive 时真实标签数被错误显示为零的问题。

### 2.4 修复 few-shot support 生成与隔离

新增了可测试的 support 生成逻辑：

- 支持 `--profile/--shot/--seed/--data-root/--output-root`。
- 每个 seed 生成独立的 `meta.txt`、类别列表、support 标签和 `audit.json`。
- 保证 exact-K 选择可复算。
- 数据不足时明确失败，不静默生成错误 shot 数。
- `robust` 可限制单张图对同一类别的贡献，降低多个 support 实例集中在同一场景的风险。

### 2.5 修复 NWPU OBB 评估

当前共享 OBB 评估器已经：

- 使用固定的 NWPU 10 类顺序。
- 固定 novel 类为 `airplane`、`baseball-diamond` 和 `tennis-court`。
- 分别输出 novel、base 和 all OBB mAP@0.5。
- 无预测类别按 AP=0 纳入固定分母。
- 排除 difficult 实例后统计 GT 正样本数。
- 输出逐类 AP、预测数和 GT 数，便于审计。
- 在空预测时仍生成合法 JSON，而不是缺少结果文件。

### 2.6 服务器环境和原生扩展

服务器中已经完成：

- 在 Conda `aofs` 环境下确认 PyTorch 2.4.0+cu121 可使用 RTX 3090。
- 重新编译 `nms_rotated_ext`，并确认可从当前环境导入。
- 使用 SWIG 和 C++ 重新编译 `DOTA_devkit/_polyiou`。
- 使用相同 polygon 自交测试得到 IoU=1.0。
- 验证 339 张 evaluation 图片、标签和 `imgnamefile.txt` 文件名集合完全一致。
- 验证 evaluation 标注全部为 10 列 DOTA polygon 格式。

旋转 NMS 用于删除高度重叠的旋转预测框；polygon IoU 用于计算预测旋转多边形与真实旋转多边形的交并比。两个扩展能否正确工作会直接影响 OBB 后处理和 AP，因此不能只验证普通 Python 代码。

### 2.7 自动化测试

当前测试覆盖以下关键契约：

- profile 和环境变量解析。
- 100/500 epoch 换算。
- 周期验证和最终验证。
- resume 实验身份检查。
- exact-K support 抽样与 seed 可复现。
- support 路径隔离。
- NWPU 类别顺序和固定 mAP 分母。
- 空预测结果和零 True Positive 标签计数。
- checkpoint、manifest 和 best 指标文件。
- 三 seed 均值与样本标准差汇总。

本报告写入后重新执行：

```text
python -m unittest discover -s tests -v
Ran 47 tests in 0.969s
OK
```

当次结果为 47 项测试通过、0 failure、0 error。最终提交或发布前仍应再次执行完整测试，以发布时的输出作为最终证据。

## 3. 已完成的服务器实验及结果

### 3.1 Base momentum 对照实验

相同代码、数据和 seed 下完成了两个 100-epoch Base 对照：

| Base momentum | 独立 HBB mAP@0.5 | HBB mAP@0.5:0.95 | 结论 |
|---:|---:|---:|---|
| `0.999` | `0.0886` | `0.0322` | 长期训练后明显偏弱 |
| `0.937` | `0.2770` | `0.0979` | 当前验证较好的 Base 基线 |

`0.937` Base 的最佳 checkpoint 出现在 epoch 69，服务器路径为：

```text
/workspace/AOFS_new/runs/base/nwpu_aofs_s_m0937/weights/best.pt
```

这一结果证明论文文字与公开 YAML 的 momentum 冲突会显著影响当前数据和实现下的 Base 质量。当前项目因此把 Base 固定为 `0.937`，而 few-shot `paper` 配置继续保留论文文字报告的 `0.999`，两个阶段不再共用同一份超参数文件。

该结果只能证明“在当前代码、当前数据和当前 seed 下 `0.937` 更好”，不能证明所有数据集或所有实现中 `0.937` 必然优于 `0.999`。

### 3.2 3-shot、seed 0、500-epoch 完整探索实验

实验身份：

```text
profile=paper
dataset=nwpu
shot=3
seed=0
stage=fewtune
```

实验目录：

```text
/workspace/AOFS_new/runs/paper/nwpu_3shot_seed_0/
```

训练使用：

- 上述 `0.937` Base 的 epoch-69 `best.pt`。
- few-shot momentum `0.999`。
- 500 个物理 epoch。
- 每个 epoch 3000 个 batch，共约 150 万次 batch 迭代。
- 每 10 个 epoch 和最终 epoch 执行一次 OBB 评估。
- checkpoint 指标为 novel OBB mAP@0.5。

最佳 checkpoint：

| 项目 | 数值 |
|---|---:|
| best epoch | `39` |
| novel OBB mAP@0.5 | `0.7769436712` |
| base OBB mAP@0.5 | `0.5877821323` |
| all OBB mAP@0.5 | `0.6445305940` |
| best_fitness | `0.7769436712` |

最佳 epoch 的三个 novel 类 AP：

| Novel 类别 | AP@0.5 |
|---|---:|
| airplane | `0.8918155334` |
| baseball-diamond | `0.7669024545` |
| tennis-court | `0.6721130255` |

最终 epoch 499：

| 项目 | 数值 |
|---|---:|
| novel OBB mAP@0.5 | `0.736583` |
| base OBB mAP@0.5 | `0.639536` |
| all OBB mAP@0.5 | `0.668651` |

best novel mAP 与最终 novel mAP 相差约 `0.04036`，即 4.04 个百分点。模型后期 base/all 指标有所提高，但 novel 指标未超过早期峰值，说明 few-shot novel 类存在明显的早期峰值和后期波动，需要在新的独立 validation 协议下继续研究。

主要产物：

```text
runs/paper/nwpu_3shot_seed_0/weights/best.pt
runs/paper/nwpu_3shot_seed_0/weights/last.pt
runs/paper/nwpu_3shot_seed_0/results.csv
runs/paper/nwpu_3shot_seed_0/run_manifest.json
runs/paper/nwpu_3shot_seed_0/best_obb_metrics.json
```

### 3.3 训练耗时和性能排查

本次 3-shot 训练从 2026-07-20 持续到 2026-07-25。耗时较长的主要原因不是训练卡死，而是：

- 每个 epoch 有 3000 个 batch，500 epoch 约 150 万次 batch 迭代。
- RTX 3090 为共享 GPU，其他任务会占用显存和计算单元。
- `/workspace` 位于 FUSE 文件系统，随机读取小图片和标签时吞吐较低。
- 主训练和 support/meta DataLoader 会产生较高 CPU 与 I/O 压力。
- 每 10 个 epoch 的 OBB 验证会增加额外时间。

将约 207 MB 数据集和 support split 复制到 `/dev/shm` 后，已通过文件数量和 SHA-256 检查确认镜像一致。代表性速度从约 `3.64 it/s` 提高到约 `6.13 it/s`，但共享 GPU 竞争仍会使单 epoch 时间明显波动。

这个优化只改变数据读取位置，不改变图片、标签、support 内容、模型、损失和超参数，因此不会因为复制到内存而降低理论精度。

## 4. 当前结果的科学边界

### 4.1 为什么不能与作者论文表格直接等价

当前 `77.69%` 不能直接写成“超过作者 AOFS-L 的 65.1%”，原因包括：

1. 缺少作者 Table III 的精确 NWPU-R 数据归档和文件哈希。
2. 缺少作者三次实验实际使用的 support 样本和随机种子。
3. 当前 Base 权重由本项目重新训练，并非作者 Table III Base checkpoint。
4. 当前代码已经修复验证、抽样、类别映射、mAP 分母和恢复逻辑，行为不同于作者原始公开代码。
5. 当前为单 seed；作者表格为多次实验的 mean ± std。
6. 当前从同一个 evaluation 集的多次周期测量中选择了最高 epoch，checkpoint 选择口径与作者未知规则不同。
7. AP 插值、NMS、置信度阈值、裁剪方式和 scene split 只要有一个不同，数值就不再是同一实验协议。

因此，当前结果的正确标签是：

```text
corrected-protocol / exploratory / seed0 / evaluation-selected-best
```

它不是：

```text
strict paper reproduction
```

### 4.2 evaluation 集周期选优的问题

科学实验中应当区分：

```text
training：更新模型参数
validation：选择 epoch、超参数和 checkpoint
test/evaluation：模型完全确定后只做最终报告
```

当前实验每 10 epoch 在 evaluation 集上测试，并从约 50 次结果中选择最高的 epoch 39。虽然 evaluation 图片没有参与反向传播，但 evaluation 指标参与了“选择哪个模型”的决策，因此产生了间接数据泄漏和最大值选择偏差。

选择次数越多，越可能把随机波动当成真实泛化能力。当前 best `77.69%` 与 final `73.66%` 的差距就是需要警惕的直接信号。

后续正式论文实验应：

1. 从训练数据中建立 scene-level 隔离的 validation，不能让同一原始场景的裁剪块跨集合。
2. 只使用 validation 选择 epoch 和超参数。
3. 方法和 checkpoint 锁定后，在 test/evaluation 上测试一次。
4. 如果无法建立独立 validation，则预先固定训练轮数并报告固定 checkpoint，而不能看完 evaluation 曲线后挑最高点。

### 4.3 已核对的 Base novel 类风险

当前 Base 配置将 `airplane`、`baseball-diamond` 和 `tennis-court` 定义为 novel，Base 阶段的 loader 会按照 `cfg.base_ids` 过滤这三个 novel 类的正标注。因此目前没有发现 Base 直接使用 novel 正样本监督训练的明显泄漏。

但是，如果包含 novel 目标的图片仍出现在 Base 查询图像中，被过滤的 novel 实例可能作为未标注背景参与训练。这属于少样本检测中的 incomplete annotation 风险，后续应通过数据审计、忽略区域或实例遮挡实验验证其影响。

## 5. 这次实验现在有什么用

本次五天训练不是无效实验，其用途包括：

### 5.1 管线验收

证明从数据读取、support 生成、模型前向、损失、旋转 NMS、polygon IoU、OBB JSON 到逐类 AP 的完整链路能够运行。

### 5.2 证明模型具备学习能力

此前接近零指标不能再简单归因于 AOFS 模型无效。修正 Base、验证和评估流程后，novel 类可以获得明显的检测能力。

### 5.3 探索性基线

本次结果可以作为新方法开发阶段的内部 baseline。新模块先在相同数据、相同 Base、相同 support、相同 seed 和相同训练预算下做 A/B 筛选。

这种 A/B 只能用于决定某个想法是否值得继续，不能直接替代最终论文的独立 test 结果。

### 5.4 收敛和过拟合诊断

完整 500-epoch 曲线显示 novel 指标早期达到峰值、后期波动，而 base 指标继续变化。这为研究 support 不稳定、灾难性遗忘、Base/novel 平衡和早停策略提供了证据。

不能因为 epoch 39 在当前 evaluation 上最好，就直接把未来正式训练固定为 40 epoch；应当把“早期可能已收敛”作为假设，再用独立 validation 验证。

### 5.5 错误分析和可视化

epoch-39 `best.pt` 可以用于：

- 查看正确检测、漏检、误检和旋转角错误。
- 分析 airplane、baseball-diamond、tennis-court 的差异。
- 观察密集目标、极端方向、尺度变化和复杂背景中的失败类型。
- 形成论文方法动机和可视化案例。

### 5.6 探索性上限

`77.69%` 可以保留为 evaluation-selected oracle，表示当前数据和模型曾达到的探索性上限。报告时必须带上 oracle 标签，不能放入最终主结果栏冒充无偏测试性能。

## 6. 近期文献定位

截至 2026-07-26，目前能核实到的较新、正式发表并公开代码的少样本旋转目标检测代表工作是 FOMC：

- 论文：<https://arxiv.org/html/2403.13375>
- 代码：<https://github.com/BriFuture/fomc>

公开结果示例：

| 数据集 | 指标 | 结果 |
|---|---|---|
| DOTA | novel OBB mAP@0.5，5/10/20-shot | `25% / 34% / 49%` |
| HRSC2016 | novel OBB mAP@0.5，3/5/10-shot | `48% / 70% / 81%` |
| NWPU VHR-10 | novel HBB mAP@0.5，3/5/10-shot | `43% / 62% / 74%` |

这些数值不能互相当作统一排行榜，因为数据集、类别划分、shot 定义、框类型和评估协议不同。NWPU VHR-10 的上述 FOMC 结果是 HBB，不应与当前 NWPU-R OBB 结果直接比较。

AOFS 原论文在 NWPU-R 上报告 AOFS-L 3-shot 为 `65.1%`，但作者关键实验材料不完整，因此当前项目只能把该值作为相关工作参考，不能据此声称当前 `77.69%` 已严格超过作者。

## 7. 后续项目和论文用途

### 7.1 当前代码作为修正版基线

冻结当前不改变核心网络的版本，命名为类似：

```text
AOFS-Corrected
AOFS-Repro
```

它承担以下角色：

- 公开 AOFS 实现的可运行基线。
- 统一数据、训练和评估协议。
- 自己新方法的主要消融起点。
- 所有新方法共用的 Base checkpoint 和评估器。

### 7.2 自己的方法方向

当前选择的是“平稳型方法改进”：创新点明确，但控制实现风险、训练成本和新增参数。

暂定推荐方向为：

> 方向感知的可靠支持原型调制。

候选组成包括：

- 为同一个 support 实例生成少量旋转和尺度视图。
- 计算多视图 support 特征一致性。
- 根据一致性对 support 原型或 AOFS 动态权重进行可靠度加权。
- 使用轻量 EMA 类别原型降低单个 support 样本和单个 batch 带来的波动。
- 可选地加入小权重的一致性损失。

这一方向尚未完成正式设计，也尚未写入训练代码。后续必须先检索相近方法、冻结设计、定义消融和公平协议，再开始实现。

### 7.3 最终论文的比较结构

最终主实验至少应包含：

| 实验 | 目的 |
|---|---|
| 作者公开实现或尽可能接近的 runnable baseline | 说明公开代码行为 |
| AOFS-Corrected | 统一、可信的修正版基线 |
| AOFS-Corrected + support 可靠度 | 验证可靠度加权贡献 |
| AOFS-Corrected + 原型稳定化 | 验证 EMA/稳定原型贡献 |
| 完整方法 | 验证组合效果 |

所有方法必须使用：

- 相同 Base checkpoint。
- 相同训练/validation/test 划分。
- 相同 support 文件。
- 相同 seed。
- 相同训练预算和 checkpoint 规则。
- 相同 OBB 评估器。

### 7.4 推荐实验阶段

阶段一：低成本方法筛选

```text
3-shot / seed 0 / 独立 validation
AOFS-Corrected vs 单个新模块
```

只有稳定提升且没有明显训练异常的方法才进入下一阶段。

阶段二：小规模确认

```text
3-shot / 3 seeds
baseline vs 完整方法
```

检查提升是否超过随机波动，并计算 mean ± sample std。

阶段三：正式矩阵

```text
3/5/10-shot × 3 seeds
baseline 与完整方法
```

阶段四：扩展数据集

至少增加 DOTA 或 DIOR-R 中一个公开 OBB 数据集。只有单一 NWPU 数据集通常不足以证明方法具有跨数据集泛化能力。

### 7.5 论文中允许和不允许的表述

允许：

> 我们基于 AOFS 公开代码构建了经过审计的可复现实验基线，并在统一协议下验证所提方法的改进。

> 初步 exploratory 运行表明修正版流程能够学习 novel 类，但该运行使用 evaluation 选优，因此不作为最终主结果。

不允许：

> 我们严格复现了作者 Table III。

> 当前 77.69% 已证明超过 AOFS-L 65.1%。

> epoch 39 是无偏测试集上的最优模型。

除非以后获得作者完整实验材料并完成同协议验证，否则不能使用上述严格复现或直接超越的结论。

## 8. 下一步工作清单

1. 在服务器记录当前源代码、Base、best、last、manifest 和指标文件的 SHA-256。
2. 将当前 3-shot 运行标记为 `exploratory/evaluation-selected-oracle`。
3. 建立 scene-level 隔离的 train/validation/test 协议。
4. 确认能否增加 DOTA 或 DIOR-R 数据集。
5. 完成“方向感知的可靠支持原型调制”相关工作检索和方法设计。
6. 先运行 3-shot seed 0 的短周期 baseline/新方法筛选。
7. 方法冻结后运行 3/5/10-shot、三个 seed。
8. 输出 mean、sample std、novel/base/all mAP、逐类 AP、训练时间和参数量。
9. 保存全部失败实验和负结果，不只保留最高指标。
10. 持续更新 `MODIFICATIONS_FROM_UPSTREAM.md`，明确区分上游、兼容修复、评估修复和自己的方法。

## 9. 一句话总结

最近工作的核心成果不是“已经复现或超过了作者的论文数值”，而是找出了此前低性能和不可靠评估的主要原因，建立了能够完整训练、恢复、验证和审计的 AOFS 修正版基线，并完成了一次 500-epoch 探索实验；这次结果将用于验证管线、分析过拟合、筛选新方法和构建后续论文基线，而正式论文性能仍需在独立 validation/test、统一协议和多 seed 条件下重新获得。
