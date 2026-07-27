# CLAUDE.md

## 项目概述

**AOFS (Arbitrary Oriented Few-Shot Object Detection in Remote Sensing Images)** 是一个面向遥感图像的任意方向（旋转框）小样本目标检测框架。

| 项目信息 | 详情 |
|----------|------|
| 论文 | "Arbitrary Oriented Few-Shot Object Detection in Remote Sensing Images" |
| 作者 | Quanjun Chen 等 |
| 发表 | IEEE TGRS, 2024 |
| 代码仓库 | https://github.com/lalalayong/AOFS |
| 基础框架 | YOLOv5 (ultralytics/yolov5) |
| 许可 | GPL-3.0 |

---

## 一、论文核心思想

### 1.1 研究问题
遥感图像中，新类别目标标注样本极少（如仅3、5、10个实例），且目标呈任意旋转角度。传统目标检测器难以同时应对"小样本"和"旋转框"两个挑战。

### 1.2 核心贡献

1. **AFM（Attention Feature Modulation）模块**：通过元学习分支（Reweight网络）为每个类别动态生成卷积核权重，实现类别特定的特征调制。解决了小样本场景下新类别特征表达不足的问题。

2. **CSL（Circular Smooth Label）角度编码**：将连续角度值编码为180个类别的圆形平滑标签（高斯核编码），将角度回归问题转化为分类问题，避免了角度周期性（边界不连续）带来的训练困难。

3. **两阶段训练策略**：
   - **阶段一（Base Training）**：在丰富标注的基类数据上训练检测器 + 元网络
   - **阶段二（Few-Shot Tuning）**：仅在 K-shot 新类样本上微调，元网络为新类生成动态卷积核

### 1.3 适用数据集
- NWPU-R（10类，novel=3类）
- DIOR-R（20类，novel=5类）
- DOTA（16类，通过 novel 分片机制支持）

---

## 二、模型结构

### 2.1 双网络架构

AOFS 由两个并行子网络组成：

```
主检测网络 (AOFS_s/l.yaml)          元网络/Reweight (reweight_s/l.yaml)
     │                                        │
     3ch RGB 输入                              4ch (RGB + Mask) 输入
     │                                        │
     CSPDarknet Backbone                       Conv → MaxPool → ... 
     (Conv → C3 → SPPF)                       (小型卷积网络)
     │                                        │
     FPN + PAN Neck                           GlobalMaxPool2d × 3
     │                                        │
     3× DynamicConv2d ←─── 动态卷积核 ────────┘ (P3/P4/P5 各一个)
     │
     Detect Head (nc + 5 + 180 输出/锚点)
```

### 2.2 关键模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **Model** | `models/yolo.py:103` | 双网络组装入口，`Model(cfg1, cfg2)` |
| **Detect** | `models/yolo.py:41` | 旋转框检测头，输出 `nc + 5 + 180` |
| **DynamicConv2d** | `models/dynamic_conv.py:144` | 类别条件动态卷积 |
| **GlobalMaxPool2d** | `models/pooling.py:8` | 全局最大池化（元网络输出层） |
| **ComputeLoss** | `utils/loss.py:91` | 四部分联合损失 |

### 2.3 Detect 头输出格式（每个锚点）

```
[cx, cy, l, s, obj_conf, cls_0, ..., cls_nc-1, θ_0, ..., θ_179]
├── 中心偏移 (×2)
├── 长边/短边 (×2)  
├── 目标置信度 (×1)
├── 类别 logits (×nc)
└── 角度 CSL 编码 (×180)  ← 高斯软标签，σ=2.0，覆盖[-90°, 90°)
```

### 2.4 前向传播流程

```python
# model.forward(imgs, metax, mask)
dynamic_weights = model.meta_forward(metax, mask)  # 元网络 → 3个卷积核
x = model._forward_once(imgs, dynamic_weights)     # 检测网络 + 动态卷积
# _forward_once 内部：每个 DynamicConv2d 后执行类别特征平均融合
```

---

## 三、代码目录结构

```
AOFS-main/
├── train.py                  # 训练入口（基础训练 + few-shot微调）
├── val.py                    # 验证（先计算动态权重原型，再检测）
├── detect.py                 # 推理（同验证流程）
├── dataset.py                # listDataset（基础数据）+ MetaDataset（元学习数据）+ build_dataset()
├── image.py                  # 图像加载、HSV增强、多边形掩码填充
├── cfg.py                    # 全局配置（类别划分、novel/base、tuning状态）
├── util.py                   # IoU、NMS、read_data_cfg 等工具函数
│
├── models/
│   ├── yolo.py               # ★ Model类、Detect头、parse_model()、parse_model_2()
│   ├── common.py             # Conv、C3、SPPF、DetectMultiBackend 等通用模块
│   ├── dynamic_conv.py       # ★ DynamicConv2d（类别条件动态卷积）
│   ├── pooling.py            # GlobalMaxPool2d、GlobalAvgPool2d
│   ├── experimental.py       # CrossConv、MixConv2d、Ensemble、attempt_load()
│   ├── tf.py                 # TensorFlow/Keras 等价导出
│   ├── AOFS_s.yaml           # ★ 小型检测器模型配置
│   ├── AOFS_l.yaml           # ★ 大型检测器模型配置
│   ├── reweight_s.yaml       # ★ 小型元网络模型配置
│   └── reweight_l.yaml       # ★ 大型元网络模型配置
│
├── cfg/
│   ├── fewyolov5_nwpu.data   # NWPU基础训练配置
│   ├── fewyolov5_dior.data   # DIOR基础训练配置
│   ├── fewtunev5_nwpu_10shot.data  # NWPU 10-shot微调配置
│   └── fewtunev5_dior_10shot.data  # DIOR 10-shot微调配置
│
├── data/
│   ├── nwpu_poly.yaml        # NWPU 数据集定义（10类）
│   ├── dior_poly.yaml        # DIOR 数据集定义（20类）
│   ├── nwpu_novels.txt       # NWPU novel类别（逗号分隔，1行）
│   ├── dior_novels.txt       # DIOR novel类别（逗号分隔，5种分片）
│   └── hyps/
│       ├── hyp.finetune_nwpu.yaml  # NWPU超参数
│       └── hyp.finetune_dior.yaml  # DIOR超参数
│
├── utils/
│   ├── loss.py               # ★ ComputeLoss（CIoU + BCEobj + BCEcls + BCEtheta）
│   ├── rboxs_utils.py        # ★ CSL编码、poly↔rbox转换、角度正则化
│   ├── metrics.py            # AP计算、CIoU、混淆矩阵
│   ├── augmentations.py      # HSV增强、透视、Mosaic、MixUp
│   ├── datasets.py           # 标准数据加载器（create_dataloader）
│   ├── general.py            # non_max_suppression_obb、各类工具函数
│   ├── nms_rotated/          # C++/CUDA旋转框NMS扩展
│   │   └── src/              # .cpp/.cu/.h 源码
│   ├── torch_utils.py        # ModelEMA、select_device 等
│   └── plots.py              # 可视化
│
├── DOTA_devkit/              # DOTA评估工具包
│   ├── dota_evaluation_task1.py  # ★ OBB mAP评估（多边形IoU）
│   ├── ImgSplit_multi_process.py # 大图切分（并行）
│   ├── ResultMerge_multi_process.py # 切分结果合并+NMS
│   └── polyiou.py/polyiou.cpp    # C++多边形IoU
│
├── tools/                    # 数据集准备脚本
│   ├── gen_fewlist_nwpu.py   # 构建NWPU小样本列表
│   ├── gen_fewlist_dior.py   # 构建DIOR小样本列表
│   ├── gen_dict_file.py      # 生成训练字典（如 traindict_bbox_10shot.txt）
│   ├── label_nwpu.py         # DOTA→YOLO标签转换（NWPU）
│   └── label_1c_nwpu.py      # 每类单独标签（AFM模块依赖）
│
├── data/scripts/download_weights.sh  # 权重下载脚本
└── requirements.txt          # Python依赖
```

---

## 四、训练流程

### 4.1 数据准备（一次性）

```bash
# 1. 切分大图（DOTA格式需要）
python DOTA_devkit/ImgSplit_multi_process.py

# 2. 生成标签
python tools/label_nwpu.py $DATA_PATH    # 全量标签 + 每类标签

# 3. 生成每类标签（AFM模块需要）
python tools/label_1c_nwpu.py $DATA_PATH

# 4. 生成少样本数据集
python tools/gen_fewlist_nwpu.py

# 5. 生成训练字典
python tools/gen_dict_file.py $DATA_PATH nwpu
```

### 4.2 阶段一：基础训练

```bash
python train.py \
  --cfg1 models/AOFS_s.yaml \
  --cfg2 models/reweight_s.yaml \
  --data data/nwpu_poly.yaml \
  --cfgdata cfg/fewyolov5_nwpu.data \
  --hyp data/hyps/hyp.finetune_nwpu.yaml \
  --epochs 100 --batch-size 1 --img 1024 --device 0 --noval
```

**关键行为：** `cfg.tuning = False`，`cfg.base_classes = 全部类别 - novel类别`，在基类数据上训练双网络。

### 4.3 阶段二：Few-Shot 微调

```bash
python train.py \
  --weights weights/base_best.pt \
  --data data/nwpu_poly.yaml \
  --cfgdata cfg/fewtunev5_nwpu_10shot.data \
  --hyp data/hyps/hyp.finetune_nwpu.yaml \
  --batch-size 1 --img 1024 --device 0 --noval
```

**关键行为：** `cfg.tuning = True`，`cfg.shot` 从 meta 文件名解析，少样本数据通过 `repeat=100` 放大，从 epoch 0 重新开始。

### 4.4 验证与评估

```bash
# 生成检测结果 JSON
python val.py --weights weights/tune_best.pt \
  --data data/nwpu_poly.yaml --cfgdata cfg/fewtunev5_nwpu_10shot.data \
  --batch-size 1 --img 1024 --task val --device 0 --save-json \
  --name tune_nwpu_10shot_AOFS_s_run1_split

# DOTA Task 1 评估
python DOTA_devkit/dota_evaluation_task1.py \
  --base_path runs/val/tune_nwpu_10shot/.../tune_best \
  --annopath $DATA_ROOT/evaluation/labelTxt/{:s}.txt \
  --imagesetfile $DATA_ROOT/imgnamefile.txt
```

---

## 五、关键参数配置

### 5.1 模型参数

| 参数 | AOFS_s | AOFS_l | 说明 |
|------|--------|--------|------|
| depth_multiple | 0.33 | 1.0 | 网络深度缩放因子 |
| width_multiple | 0.50 | 1.0 | 通道宽度缩放因子 |
| nc | 80 | 80 | 默认类别数（运行时覆盖） |
| no (输出/锚点) | nc + 185 | nc + 185 | = nc + 5 + 180 |

### 5.2 训练超参数（hyp.finetune_*.yaml）

| 参数 | 值 | 说明 |
|------|-------|------|
| lr0 | 0.001 | 初始学习率 |
| momentum | 0.937 | SGD 动量 |
| weight_decay | 0.0005 | 权重衰减 |
| box | 0.05 | 框回归损失权重（CIoU） |
| cls | 0.5 | 分类损失权重 |
| theta | 0.5 | 角度分类损失权重（CSL） |
| obj | 1.0 | 目标置信度损失权重 |
| degrees | 180.0 | 随机旋转增强范围 |
| mosaic | 0.75 | Mosaic 增强概率 |
| cls_theta | 180 | CSL 角度分类数 |
| csl_radius | 2.0 | CSL 高斯半径 (σ) |

### 5.3 损失函数构成

```
总损失 = 0.05 × CIoU_box + 1.0 × BCE_obj + 0.5 × BCE_cls + 0.5 × BCE_theta
```

其中 BCE_theta 使用高斯软标签（CSL），与多个相邻角度 bin 进行比较，避免硬标签的边界不连续问题。

### 5.4 数据集配置参数

| 参数 | 基础训练 | Few-Shot微调 |
|------|---------|-------------|
| tuning | 0 | 1 |
| novelid | 0 | 0 |
| repeat | (无) | 100 |
| max_epoch | (默认) | 50000 |
| meta | traindict_full.txt | traindict_bbox_{K}shot.txt |

---

## 六、目前已发现的问题

### 6.1 🔴 阻塞级问题

1. **缺少 `models/lsknet.py`**：`models/yolo.py:33` 有 `from models.lsknet import *`，该文件不存在。如果代码路径触发此 import，会导致 ImportError。确认是否使用了 LSKNet backbone 变体——如未使用，可安全注释。

2. **`train.py` 验证代码被注释掉**：`train.py:267-283` 的 `val_loader`、`metaset_val` 创建代码完全被注释，但 `train.py:429-441` 的验证调用仍引用这些变量。运行时若 `--noval` 未设置（即进行验证），会因 `NameError: val_loader is not defined` 崩溃。

### 6.2 🟡 警告级问题

3. **遗留 `pdb.set_trace()` 断点**：以下 11 个文件中存在 `pdb.set_trace()` 调试断点，代码执行到此处会暂停：
   - `models/dynamic_conv.py`
   - `util.py`
   - `image.py`
   - `dataset.py`
   - `DOTA_devkit/dota_evaluation_task1.py`
   - `DOTA_devkit/hrsc2016_evaluation.py`
   - `DOTA_devkit/mAOE_evaluation.py`
   - `DOTA_devkit/ucasaod_evaluation.py`
   - `DOTA_devkit/ImgSplit_multi_process.py`
   - `DOTA_devkit/ResultMerge_multi_process.py`
   - `DOTA_devkit/ResultEnsembleNMS_multi_process.py`

4. **硬编码路径 `topath()`**：`dataset.py:20` 的函数将所有路径中的 `scratch` 替换为 `tmp_scratch/basilisk`，这是原作者机器特有的路径映射，在任何其他机器上会导致文件找不到。

5. **`train.py` 中 `variables_never_assigned_in_this_block_but_is_used` 隐式依赖**：`train.py:63-66` 使用解包语法读取 datacfg 中的多个字段，但依赖的是较旧版 YOLOv5 的 `read_data_cfg` 返回格式，其键名可能与当前代码版本不一致。

### 6.3 🔵 代码质量问题

6. **大量注释掉的代码**：`train.py`、`models/yolo.py`、`models/reweight_s.yaml` 中存在大量被注释而非删除的代码块（超过 100 行），降低可读性。

7. **`dynamic_conv.py` 中的孤立代码**：第 80-141 行有一个完整的 `dynamic_conv2d` 工厂函数被注释掉，与当前使用的类版本功能重复。

8. **死代码引用**：`cfg.py` 和 `dataset.py` 中存在对 'coco' 数据集的引用，但项目实际只支持 DOTA、NWPU、DIOR 三种数据集。

9. **`export.py` / `hubconf.py` 与双网络架构不兼容**：标准 YOLOv5 导出逻辑假定单网络模型，AOFS 的双网络（Model + Reweight）无法直接导出为 ONNX/TensorRT 等格式。

10. **`models/__init__.py` 为空文件**：虽然不影响代码运行（Python 仍可将其作为包导入），但缺少 `__init__.py` 中的显式导出声明使得包结构不够清晰。

---

## 七、后续修改代码时的注意事项

### 7.1 必须遵循的代码约定

1. **双网络必须保持同步**：修改 `AOFS_s/l.yaml` 的层结构（特别是 DynamicConv2d 的数量和 channel 维度）时，必须同步修改 `reweight_s/l.yaml` 中对应层的 channel 输出，使动态卷积核维度匹配。三个 DynamicConv2d 分别对应 P3(256ch)、P4(512ch)、P5(1024ch)。

2. **模型加载使用 `intersect_dicts`**：`train.py:145` 使用 `intersect_dicts` 进行部分权重匹配。修改模型结构后，需要确认新增/删除的层能正确处理（strict=False 模式）。

3. **Detect 头输出维度**：`self.no = nc + 5 + 180`（`models/yolo.py:48`）。如果修改 `cls_theta`（角度分类数）或添加/移除输出项，必须同步修改此处、`ComputeLoss.build_targets` 中的目标构建、以及 `utils/general.py` 中 `non_max_suppression_obb` 的角度解码逻辑。

4. **CSL 角度编码一致性**：角度编码/解码涉及三个位置，修改参数时必须三者同步：
   - `rboxs_utils.py:9` `gaussian_label_cpu`（编码，180 bin）
   - `utils/loss.py:171-172` 角度损失计算（class_index = 5 + nc）
   - `utils/general.py` `non_max_suppression_obb`（解码，`theta = (pred_class - 90) / 180 * pi`）

5. **cfg.py 全局状态**：`cfg.py` 使用模块级可变全局状态（`__C` edict）。`train.py`、`val.py`、`dataset.py` 都通过 `from cfg import cfg` 共享同一状态。修改配置逻辑时注意这种隐式耦合。

6. **MetaDataset 数据流**：`MetaDataset` 的输出（4通道：RGB+Mask）输入到 Reweight 网络。如果修改 Reweight 网络的输入通道数（`reweight_s.yaml` 的 `ch: 4`），必须同步修改 `MetaDataset.get_img_mask()` 和 `meta_forward()` 中的拼接逻辑。

### 7.2 建议优先修复的安全问题

1. 移除所有 `pdb.set_trace()` 调用（11 处）
2. 删除或参数化 `topath()` 函数（`dataset.py:20-21`）
3. 修复 `train.py` 验证代码：要么取消注释 `val_loader` 创建部分，要么在验证调用处添加条件保护

### 7.3 开发建议

- **优先使用 `--noval` 标志训练**以避免 `train.py` 中验证代码的 bug
- 在修改 `models/yolo.py` 前先确认 `models/lsknet.py` 是否需要
- 测试 on 新环境时，将 `dataset.py` 中的 `topath()` 改为恒等函数或完全移除
- 模型导出（ONNX/TensorRT）在当前架构下不直接支持，如有需要必须重写 `export.py` 以处理双网络前向传播

### 7.4 文件依赖关键路径

```
train.py
  → cfg.py (全局配置)
  → dataset.py (listDataset, MetaDataset, build_dataset)
  → models/yolo.py (Model → parse_model + parse_model_2)
  → models/dynamic_conv.py (DynamicConv2d)
  → models/pooling.py (GlobalMaxPool2d)
  → utils/loss.py (ComputeLoss + CSL角度损失)
  → utils/rboxs_utils.py (CSL编码 + poly↔rbox转换)
  
val.py / detect.py
  → 同上 + utils/general.py (non_max_suppression_obb)
  → utils/nms_rotated/ (C++旋转框NMS)
```

---

## 八、角色定位与协作原则

### 8.1 我的角色

| 角色 | 职责 |
|------|------|
| 论文复现助手 | 帮助理解论文方法和作者代码，忠实还原实验流程 |
| 代码审查助手 | 检查代码问题、兼容性、潜在风险 |
| 环境兼容性分析助手 | 分析本地/服务器环境差异，提供适配建议 |
| 实验记录助手 | 协助记录实验过程、结果和偏离分析 |

### 8.2 核心原则

1. **忠实复现优先**：目标是还原作者的代码和实验结果，而非自由重构或重新设计。任何针对代码逻辑或超参数的修改都必须有明确依据（论文、作者官方说明、或确凿的 bug 证据）。

2. **最小化修改**：只在必要时修改代码，修改变动范围尽可能小。优先修复阻塞运行的问题，功能性的改动需经明确授权。

3. **先核对、再修改**：修改逻辑前需比对论文描述与代码实现，发现差异时先标记、确认后再决定是否修改。

4. **保持可追溯性**：每次修改需说明原因（引用论文章节 / 作者 README / 错误日志）。所有对代码的修改都要有据可查。

### 8.3 流程

```
阅读论文与官方说明
        ↓
分析作者原始代码
        ↓
核对论文与代码差异 → 记录差异，标记需确认项
        ↓
本地进行最小化修改 → 每次修改附原因
        ↓
本地静态检查与小规模测试
        ↓
上传服务器
        ↓
采集服务器环境信息
        ↓
服务器小规模运行测试
        ↓
正式训练
        ↓
评估论文指标
        ↓
分析与论文结果的差异 → 记录偏离及可能原因
```

---

## 九、代码修改规范

### 9.1 修改权限

| 类型 | 权限 | 说明 |
|------|------|------|
| 修复 ImportError / SyntaxError | ✅ 允许 | 缺失文件、语法错误等阻塞运行的 bug |
| 移除 `pdb.set_trace()` | ✅ 允许 | 调试断点会阻塞训练 |
| 修正环境路径硬编码 | ✅ 允许 | 如 `topath()` 等机器相关路径 |
| 补全依赖项 | ✅ 允许 | `requirements.txt` 中遗漏的包 |
| 修复已注释代码导致的 NameError | ✅ 允许 | 如 `train.py` 的 val_loader 问题 |
| 清理注释掉的无用代码 | ⚠️ 需授权 | 改善可读性，但不影响运行 |
| 修改损失函数逻辑 | ⚠️ 需授权 | 核心逻辑，需先确认是否为 bug |
| 调整超参数 | ⚠️ 需授权 | 除非明显错误（如 lr=0），否则保持原值 |
| 修改网络结构 | ⚠️ 需授权 | 核心架构，需论文依据 |
| 重构代码架构 | ❌ 禁止 | 违反忠实复现原则 |

### 9.2 修改记录格式

每次修改代码时，记录以下信息：

```
[修改日期] [文件路径] [修改类型: Fix / Adapt / Cleanup]
  原因: <为什么需要改，引用的依据>
  内容: <改了什么，before/after>
  影响: <影响了哪些功能 / 流程>
```

我会在每次修改后整理这些信息。

### 9.3 禁止的修改

- **不要重写现有功能**：即使某个实现看起来"不够好"，也不要重新设计
- **不要"顺便"修改**：不要在一个修复中夹带无关的改动
- **不要在未确认差异前修改核心逻辑**：论文说 A 代码写成 B → 先标记、讨论确认 → 再决定

---

## 十、活动日志与实验记录

### 10.1 活动日志

每次与本项目的交互应记录到 `ACTIVITY_LOG.md`（项目根目录），包括：

- 做了什么（新增修改 / 环境配置 / 调试 / 训练）
- 为什么这样做
- 改动了哪些文件（具体到文件路径和关键行号）
- 结果是什么
- 下一步计划

**格式要求：**
- 按时间倒序（最新在最前面）
- 每条记录包含：时间戳、类型标签、摘要、详情

### 10.2 实验记录

正式训练开始后，应将每次完整的训练运行记录到 `experiments/` 目录下的单独文件，包括：

| 项目 | 内容 |
|------|------|
| 实验名称 / ID | |
| 日期时间 | |
| 服务器 / GPU 型号 | |
| 完整训练命令 | |
| 数据集 / Novel 设置 | |
| 关键超参数 | |
| 训练时间 | |
| 最佳模型路径 | |
| 评估指标 | novel mAP / base mAP / overall mAP |
| 与论文报告值的差异 | |
| 差异可能原因 | |
| 备注 | |

---

## 十一、环境管理规范

### 11.1 服务器信息记录

连接到新服务器时，应采集并记录以下环境信息：

- 操作系统和版本
- CPU 型号和核心数
- GPU 型号、数量和显存
- CUDA 版本 (`nvcc --version`)
- NVIDIA 驱动版本 (`nvidia-smi`)
- Python 版本
- PyTorch 版本及其 CUDA 版本 (`torch.version.cuda`)
- 已安装的关键包版本
- 磁盘可用空间

以上信息保存在 `ENVIRONMENT.md` 中（本地和服务器各一份）。

### 11.2 环境差异处理

本地（Windows）和服务器（通常 Linux）之间的环境差异需要特别关注：

- 文件路径分隔符（`/` vs `\`）
- 文件系统大小写敏感性
- C++ 扩展编译（`.pyd` vs `.so`）
- 多进程启动方式（`spawn` vs `fork`）
- CUDA 架构兼容性（`sm_xx` 编译目标）

---

## 十二、结果分析规范

### 12.1 与论文对比

正式训练完成后，对比论文报告的结果：

- Novice mAP 绝对值差异
- Base mAP 绝对值差异
- 不同 shot 数量下的趋势是否一致
- 如果结果显著低于论文 (>&gt;2 mAP)，分析可能原因：
  - 训练是否充分收敛
  - 数据预处理是否与作者一致
  - 随机种子是否固定
  - 评估协议是否匹配

### 12.2 偏离记录

任何与论文/作者官方说明不一致的操作都应记录在 `DEVIATIONS.md` 中，包括：

- 无法完全复现的步骤
- 使用的替代方案
- 参数调整及原因
- 预期影响评估

---

## 十三、沟通规范

### 13.1 我需要主动汇报的情况

- 发现一个可能阻塞运行的问题
- 发现代码逻辑与论文描述不一致
- 完成了一组修改，需要总结
- 训练完成，需要报告结果
- 遇到不确定如何处理的决策点

### 13.2 应等待确认的情况

- 需要修改网络结构
- 需要调整超参数（除非是明显的致命错误）
- 需要安装大型依赖（>500MB）
- 需要删除或移动大量文件
- 评估结果与论文显著不符，需要决定下一步

### 13.3 主动给出判断和建议的情况

- 明显且容易修复的 bug（附带修复理由）
- 环境兼容性问题（附带适配方案）
- 代码质量问题（附带影响评估）
- 实验结果分析（附带可能原因）

---

## 十四、快速参考：核心操作规则

> 🔗 完整规范见 [`docs/REPRODUCTION_RULES.md`](docs/REPRODUCTION_RULES.md)

### 14.1 保护原则（三不）

- ❌ **不擅自修改**：模型结构、损失函数、超参数、数据划分 → 必须先报告，等确认
- ❌ **不编造缺失代码**：文件缺失 → 先搜索 README/gitmodules/发布页，不自行实现
- ❌ **不隐藏差异**：论文与代码不一致 → 记录差异表，不悄悄选一个

### 14.2 修改前必须回答

涉及源码的任何修改前，先报告：
```
问题级别（P0-P4）/ 类型（1-7）/ 涉及文件 / 推荐方案 / 是否影响论文指标 / 回退方法
```
详细检查清单见 `REPRODUCTION_RULES.md` 第八节。

### 14.3 问题优先级速查

| 级别 | 含义 | 我的动作 |
|------|------|---------|
| P0 | 安全/数据风险 | 🛑 立即停止 |
| P1 | 阻塞运行 | 报告 + 提出方案 + 等确认 |
| P2 | 可能影响结果 | 记录 + 标注 + 等确认 |
| P3 | 兼容性 | 记录 + 建议 |
| P4 | 代码质量 | 记录但通常不修改 |

### 14.4 安全管理

- 严禁：`sudo`、`rm -rf`、格式化、杀进程、覆盖数据、执行未知脚本
- 禁止将密码/Token/密钥写入任何文件或聊天
- 破坏性操作必须先报告：做什么 / 影响 / 耗时 / 回退方式

### 14.5 测试顺序

```
静态检查 → 导入测试 → Smoke Test → 短训练 → 完整训练
```
每级通过后才进入下一级。完整训练需授权。

### 14.6 记录文件

| 文件 | 用途 | 何时写 |
|------|------|--------|
| [`docs/modification_log.md`](docs/modification_log.md) | 源码修改 | 每次修改后 |
| [`docs/experiment_log.md`](docs/experiment_log.md) | 实验运行 | 每次运行后（含失败） |
| [`docs/reproduction_notes.md`](docs/reproduction_notes.md) | 长期笔记 | 持续更新 |
| [`docs/environment_comparison.md`](docs/environment_comparison.md) | 环境对比 | 首次连接服务器 |
| [`docs/known_issues.md`](docs/known_issues.md) | 问题追踪 | 发现问题时 |

### 14.7 项目状态

- **Git**：项目未建立 Git 管理。建议在修改前 `git init` 并创建 `reproduction/minimal-fixes` 分支
- **环境**：本地 Windows 11，服务器（待确定）
- **当前阶段**：代码分析完成，问题已识别 → 等待开始最小修改
