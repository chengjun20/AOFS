# 复现笔记

> 长期复现信息的汇总。包含已确认事实、待确认问题、重要决策及原因。
> 不确定的内容必须标注状态：[已确认] / [待确认] / [推测] / [缺少证据]

---

## 论文核心任务

任意方向的遥感图像小样本目标检测（Oriented Few-Shot Object Detection）。

- 输入：遥感图像（RGB）
- 输出：旋转边界框（OBB），格式 [cx, cy, l, s, θ]，θ ∈ [-π/2, π/2)
- 角度处理：CSL (Circular Smooth Label)，180 类高斯编码
- 少样本策略：二阶段训练（Base Training → Few-Shot Tuning）

---

## 数据集

### NWPU-R
- 类别：10 类
- Novel 划分：airplane, baseball-diamond, tennis-court（共 3 类）[已确认]
- Base 划分：ship, storage-tank, basketball-court, ground-track-field, harbor, bridge, vehicle（共 7 类）[已确认]
- 标注格式：DOTA（8 点多边形）
- 状态：配置文件已存在，数据待下载

### DIOR-R
- 类别：20 类
- Novel 划分：airplane, baseballfield, tenniscourt, trainstation, windmill（novelid=0）[已确认]
- Base 划分：其余 15 类
- 标注格式：DOTA（8 点多边形）
- 状态：配置文件已存在，数据待下载

---

## 模型结构要点

- 双网络：AOFS 检测网络 + Reweight 元网络 [已确认]
- DynamicConv2d 数量：3 个（P3/256ch、P4/512ch、P5/1024ch）[已确认]
- 检测头输出：nc + 5 + 180 = nc + 185 [已确认]
  - 其中 180 = cls_theta（角度 CSL 分类数）[已确认]
- CSL 高斯半径（csl_radius）：作者超参数文件中为 2.0 [已确认]
  - 注意：代码中 gaussian_label_cpu 默认值为 sig=4.0 [待确认]

---

## 训练设置

### 基础训练
- epochs：100
- batch_size：1
- img_size：1024
- 优化器：SGD（lr=0.001, momentum=0.937）
- 论文是否明确：待确认

### Few-Shot 微调
- max_epoch：50000（实际 epochs = max_epoch / repeat = 500）
- repeat：100
- shot 数从 meta 文件名解析
- 论文是否明确：待确认

---

## 论文与代码差异

| 项目 | 论文 | 代码/配置 | 状态 |
|------|------|----------|------|
| csl_radius | 待确认 | 超参文件=2.0, 代码默认=4.0 | 待确认 |
| base training epochs | 待确认 | 100 | 待确认 |
| 训练用的 loss 权重 | 待确认 | box=0.05, cls=0.5, theta=0.5, obj=1.0 | 待确认 |
| 角度损失类型 | 待确认 | BCEWithLogitsLoss + CSL 高斯软标签 | 待确认 |

---

## 环境

### 作者目标环境（来自 README）
- Python 3.7+
- PyTorch >= 1.7（推荐 1.10.1）
- CUDA 9.0+

### 本地环境
- 操作系统：Windows 11 Home China
- Python：(待采集)
- PyTorch：(待安装)
- CUDA：(待安装)

### 服务器环境
- (待采集)

---

## 已确认结论

1. 项目基于 YOLOv5 (ultralytics/yolov5)，已做大量定制修改 [已确认]
2. 使用 DOTA_devkit 进行多边形 IoU 评估、图像切分和结果合并 [已确认]
3. 类别划分通过 cfg.py 的全局 edict 状态管理 [已确认]
4. 模型不直接支持 ONNX/TensorRT 导出（双网络架构）[已确认]

---

## 待确认问题

1. 论文中 csl_radius 的准确值是什么？代码默认 4.0 vs 配置文件 2.0 [缺少证据]
2. 论文报告的 base training 是否也是 100 epochs？[需要论文确认]
3. LSKNet backbone 是否为可选用法？是否需要补充该文件？[待确认]
4. 作者提供的预训练权重 base_best.pt 是否可用？[需要下载验证]
5. 论文中各类别的 mAP 基准值（后续用于对比）[需要论文确认]

---

## 重要决策及原因

(暂无)

---

## 当前复现进度

- [x] 论文理解
- [x] 代码结构分析
- [x] 问题识别（10 个已知问题已记录在 known_issues.md）
- [ ] 环境搭建（本地）
- [ ] 环境搭建（服务器）
- [ ] 阻塞问题修复
- [ ] 数据准备
- [ ] 基础训练复现
- [ ] Few-Shot 微调复现
- [ ] 结果对比

---

## 下一阶段计划

1. 确认数据获取路径（NWPU-R 和 DIOR-R）
2. 搭建本地 PyTorch 1.10 测试环境
3. 修复 P1 阻塞问题
4. 逐步启动小规模 Smoke Test
