# 修改记录

> 所有源码、配置、脚本或环境适配修改的完整记录。
> 格式严格按照 REPRODUCTION_RULES.md 第十七节的模板。

---

## 修改记录索引

| 编号 | 时间 | 文件 | 类型 | 状态 | 摘要 |
|------|------|------|------|------|------|
| M-0001 | 2026-07-27 23:50 UTC+8 | dataset.py:100 | 作者代码残留 | 已完成 | 删除无条件 pdb.set_trace() 调试断点 |
| M-0002 | 2026-07-28 02:00 UTC+8 | train.py:267-271 | 作者代码残留 | 已完成 | 恢复被注释的 val_loader + metaset_val 初始化 |
| M-0003 | 2026-07-28 02:15 UTC+8 | models/common.py:30, models/yolo.py:33,392 | 文件缺失 | 已完成 | LSKNet 可选 backbone 缺失保护 |
| M-0004 | 2026-07-28 03:30 UTC+8 | image.py:221,260 | 作者代码残留 | 已完成 | 注释 fill_truth_detection 中的条件 pdb.set_trace() |

---

## M-0001

### 基本信息

- 时间：2026-07-27 23:50 UTC+8
- 操作者：Claude (复现助手)
- Git 分支：N/A（项目尚未建立 Git 管理）
- 修改前 Commit：N/A
- 修改后 Commit：N/A
- 对应问题编号：ISSUE-003
- 对应实验编号：无

### 修改对象

- 修改文件：dataset.py
- 修改函数或代码位置：load_metadict() 函数，原第 100 行
- 修改类型：作者代码残留

### 原始问题

- 问题表现：`load_metadict()` 被调用时无条件进入 pdb 交互器，阻塞训练
- 完整报错：无报错；进程在 (Pdb) 提示符处挂起
- 执行命令：任何触发 `load_metadict()` 的训练或数据加载命令
- 触发条件：`build_dataset()` 中 `dynamic=1` 或任何调用 `load_metadict()` 的路径
- 本地是否出现：是（静态分析确认）
- 服务器是否出现：待确认

### 原因分析

- 根本原因：作者调试时遗留的无条件 `pdb.set_trace()` 语句
- 判断依据：该语句位于 `load_metadict()` 函数主体中，无任何条件包裹
- 是否与作者环境差异有关：否
- 是否与依赖版本有关：否
- 是否与数据有关：否
- 是否为作者原代码问题：是
- 是否存在不确定性：否

### 修改方案

- 为什么必须修改：无条件断点会阻塞任何触发该函数的训练流程
- 为什么不能仅调整环境：pdb 行为不依赖于环境
- 为什么不能仅调整配置：pdb 行为不依赖于配置
- 采用的最小修改方案：删除该行
- 被否决的备选方案及原因：
  - 注释该行：等效于删除，但遗留无效注释
  - 改为 `if False: pdb.set_trace()`：多余

### 修改内容

修改前（原第 99–100 行）：
```python
    metadict = {line[0]: loadlines(line[1]) for line in files}

    pdb.set_trace()
    # Remove base-class images
```

修改后：
```python
    metadict = {line[0]: loadlines(line[1]) for line in files}

    # Remove base-class images
```

### 影响分析

- 是否改变模型结构：否
- 是否改变算法逻辑：否
- 是否改变输入数据：否
- 是否改变数据预处理：否
- 是否改变 Loss：否
- 是否改变 Optimizer：否
- 是否改变 Scheduler：否
- 是否改变训练流程：否（仅移除调试暂停）
- 是否改变评价流程：否
- 是否可能影响论文指标：否
- 与论文一致性：一致

### 后果与风险

- 预期后果：`load_metadict()` 执行不再暂停
- 潜在副作用：无
- 新增依赖：无
- 跨平台影响：无
- 回退方法：在新 Git 仓库中 `git checkout dataset.py`

### 验证结果

- 静态检查：`python -c "import ast; ast.parse(open('dataset.py').read()); print('Syntax OK')"` → 通过
- 导入测试：待服务器环境
- Smoke Test：待服务器环境
- 服务器测试：待执行
- 是否解决原问题：是
- 是否出现新问题：否
- 实验效果：待验证

### 结论

单行删除，风险极低。`load_metadict()` 中的无条件调试断点已移除，函数调用不再阻塞。

---

---

## M-0002

### 基本信息

- 时间：2026-07-28 02:00 UTC+8
- 对应问题编号：ISSUE-002
- 修改前/后 Commit：501610a → e68850f
- Git 分支：reproduction/minimal-fixes

### 修改对象

- 修改文件：train.py
- 修改类型：作者代码残留（被注释的代码）

### 原始问题

- 问题表现：`val_loader` 和 `metaset_val` 初始化代码被作者注释（第267-271行），但 epoch 末尾验证调用（第428-440行）仍引用这些变量，导致 `NameError: val_loader is not defined`
- 原因：调试遗留

### 修改方案

取消注释恢复作者原始代码，gs 作为第四个位置参数绑定到 stride 形参。

### 修改内容

Before（注释状态）→ After（取消注释），参数和缩进完全保留。

### 影响分析

- 是否改变模型结构/算法逻辑/输入数据/Loss/Optimizer/Scheduler：否
- 是否影响论文指标：否（恢复作者预期行为）
- 与论文一致性：一致

### 验证结果

- 静态检查：`python -m py_compile train.py` 通过
- `inspect.signature().bind()` 参数绑定通过
- 服务器测试：待执行

---

## M-0003

### 基本信息

- 时间：2026-07-28 02:15 UTC+8
- 对应问题编号：ISSUE-001
- 修改前/后 Commit：e68850f → 944d15d
- Git 分支：reproduction/minimal-fixes

### 修改对象

- 修改文件：models/common.py（注释未使用的 import）、models/yolo.py（哨兵保护 + 显式错误）
- 修改类型：文件缺失（可选 backbone 的 import 处理）

### 原始问题

- 问题表现：`models/yolo.py:33` `from models.lsknet import *` 和 `models/common.py:30` `from models import lsknet` 触发 `ModuleNotFoundError: No module named 'models.lsknet'`
- 原因：LSKNet 是论文未使用的可选 backbone，不在仓库中

### 修改方案

- `common.py`：注释未使用的 import（整个文件中无 lsknet 引用）
- `yolo.py`：`_LSKNET_MISSING` 哨兵值 + 窄范围 `ModuleNotFoundError` 捕获（仅 `exc.name == 'models.lsknet'` 时设为哨兵）
- `parse_model`：LSKNet 分支增加显式错误检查

### 影响分析

- 是否改变模型结构：否（AOFS_s/l.yaml 使用标准 CSPDarknet）
- 是否影响论文指标：否
- 与论文一致性：一致（论文无 LSKNet）

### 验证结果

- 静态检查：`python -m py_compile models/common.py models/yolo.py` 通过
- `_LSKNET_MISSING` 哨兵值 `None in {_LSKNET_MISSING}` = `False` 验证通过
- 服务器 Smoke Test：待执行

---

---

## M-0004

### 基本信息

- 时间：2026-07-28 03:30 UTC+8
- 对应问题编号：ISSUE-003
- 修改前/后 Commit：d0f01ec → 33e514f
- Git 分支：reproduction/minimal-fixes

### 修改对象

- 修改文件：image.py
- 修改类型：作者代码残留

### 原始问题

- 问题表现：`fill_truth_detection()` (line 221) 和 `fill_truth_detection_metaV2()` (line 260) 在 `ind >= n_cls or ccs[ind] >= cfg.max_boxes` 时进入 pdb 断点
- 触发条件：标签中某个基类的 bbox 数量超过配置限制（max_boxes=50）
- 是否在训练路径：是 -- 两个函数均在数据加载时被调用

### 修改方案

注释 pdb.set_trace() 并加入 `pass` 保留 if 块结构。

### 影响分析

- 是否改变模型结构/算法逻辑/输入数据/Loss/Optimizer/Scheduler：否
- 是否影响论文指标：否（仅在异常数据条件时触发）
- 与论文一致性：一致

### 验证结果

- 静态检查：`python -m py_compile image.py` 通过
- 服务器测试：待执行

---

<!-- 新修改记录请按 M-0005, M-0006 ... 追加在下方 -->

