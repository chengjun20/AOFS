# AOFS 验证标签计数修复设计

日期：2026-07-18

## 背景与根因

NWPU-R base 阶段的 1 epoch CUDA 冒烟测试完成了训练和验证，但验证汇总显示 `Images=339, Labels=0`。服务器数据审计确认：339 个标签文件包含 2114 行合法 DOTA 十列标注，缓存加载 2111 行（3 行重复标注被既有缓存逻辑去除），类别、坐标和 difficult 字段均合法。

根因位于 `val.py` 的汇总条件：代码仅在 `stats[0].any()` 为真时计算 `nt`。`stats[0]` 表示预测是否命中真实目标，因此从零训练一轮后没有 True Positive 会把 `nt` 错误设置为零，即把“没有正确预测”误报成“没有真实标签”。

## 方案选择

采用最小逻辑修复：真实标签计数始终从 `stats[3]` 独立计算；AP、Precision 和 Recall 仍只在存在 True Positive 时调用既有 `ap_per_class`。不改变模型、损失、数据过滤、类别划分、阈值或 checkpoint 指标。

未选择的方案：

- 不通过伪造一次正确预测绕过条件，因为这会污染真实指标。
- 不只修改日志文本，因为返回值和逐类标签计数也应保持正确。
- 不在本批次处理 AMP `FutureWarning`，因为它与标签统计无关且不影响当前训练正确性。

## 代码边界与数据流

在 `val.py` 增加一个纯函数，根据已拼接的 validation stats 和类别数返回固定长度的逐类真实标签计数。验证汇总先调用该函数，再单独判断是否存在 True Positive：

1. 有真实标签但零 True Positive：标签数正确，P/R/AP 保持为零。
2. 有 True Positive：标签数与现有 AP 计算都正常。
3. 完全没有 stats：返回长度为 `nc` 的全零计数，不抛出异常。

该修复只影响指标汇总与显示，不影响训练梯度或模型权重。

## 测试与验收

使用标准库 `unittest` 新增回归测试，并严格执行 RED-GREEN：

- 构造“存在真实标签、correct 全为 False”的 stats，修复前必须复现标签数错误，修复后应得到正确逐类计数。
- 构造空 stats，确认返回固定长度的全零数组。
- 运行新增测试、既有完整测试套件、`py_compile` 和 `git diff --check`。

服务器验收时从 GitHub 的 `aofs-dual-profile` 分支执行 fast-forward pull，重新运行 1 epoch smoke。验收标准是训练、验证、权重保存均完成，且验证汇总的 `Labels` 大于零；一轮训练的 P/R/AP 仍可能为零，这属于模型尚未命中的正常结果。

## 变更记录与同步

本批次修改 `val.py`，新增回归测试，并同步更新 `MODIFICATIONS_FROM_UPSTREAM.md`。本地验证、提交和推送后，服务器仅通过 Git 拉取，不直接手工编辑，以保证本地、GitHub 和服务器三处代码一致。
