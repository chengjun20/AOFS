# 环境对比

> 作者环境、本地环境、服务器环境的对比记录。
> "作者未说明" 表示该项在 README/论文/配置中未明确指定。

---

## 环境对比表

| 项目 | 作者环境 | 本地环境 | 服务器环境 | 是否一致 | 风险 | 处理建议 |
|------|----------|----------|-----------|----------|------|---------|
| 操作系统 | Linux（推测） | Windows 11 Home China | (待采集) | 待确认 | 中 | 注意路径/大小写/多进程差异 |
| Python | 3.8（推荐） | (待采集) | (待采集) | 待确认 | | |
| PyTorch | 1.10.1（推荐） | (待采集) | (待采集) | 待确认 | | |
| torchvision | 0.11.2（推荐） | (待采集) | (待采集) | 待确认 | | |
| CUDA Toolkit | 11.3（推荐） | (待采集) | (待采集) | 待确认 | | |
| cuDNN | 未说明 | (待采集) | (待采集) | 待确认 | | |
| NVIDIA Driver | 未说明 | (待采集) | (待采集) | 待确认 | | |
| GPU | 未说明 | (待采集) | (待采集) | 待确认 | | |
| 编译器 (gcc/g++) | 未说明 | (待采集) | (待采集) | 待确认 | 低 | 编译 C++ 扩展时需要 |
| SWIG | 要求安装 | (待采集) | (待采集) | 待确认 | | DOTA_devkit 多边形 IoU |
| NumPy | 未说明 | (待采集) | (待采集) | 待确认 | 低 | |
| OpenCV | 未说明 | (待采集) | (待采集) | 待确认 | 低 | |
| Cython | 未说明 | (待采集) | (待采集) | 待确认 | 低 | utils/nms_rotated |
| timm | 要求 | (待采集) | (待采集) | 待确认 | 中 | models/yolo.py import |

---

## 作者环境详情（来自 README）

```text
conda create -n Py38_Torch1.10_cuda11.3 python=3.8
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3
pip install -r requirements.txt
```

requirements.txt 内容：
```
(待解析)
```

---

## 本地环境详情

(待采集)

---

## 服务器环境详情

(待采集)

---

## 跨平台关注点

| 关注项 | 说明 | 处理方式 |
|--------|------|---------|
| 路径分隔符 | Windows `\` vs Linux `/` | 代码中使用 `/` 或 `os.path.join` |
| 大小写敏感性 | Windows 不区分，Linux 区分 | import 路径与文件名严格一致 |
| C++ 扩展 | Windows `.pyd` vs Linux `.so` | 在服务器上重新编译 |
| 多进程启动 | Windows `spawn` vs Linux `fork` | DataLoader 注意 num_workers 设置 |
| CUDA 架构 | `.pyd` 预编译 vs 源代码编译 | 服务器上从源码编译 DOTA_devkit |
| 换行符 | CRLF vs LF | 统一使用 LF |

---

<!-- 
采集命令（在服务器 Conda 环境中执行）：

echo "===== DATE =====" && date
echo "===== WORKDIR =====" && pwd
echo "===== OS =====" && uname -a
cat /etc/os-release 2>/dev/null || true
echo "===== CPU =====" && lscpu 2>/dev/null || true
echo "===== MEMORY =====" && free -h 2>/dev/null || true
echo "===== DISK =====" && df -h
echo "===== GPU =====" && nvidia-smi 2>/dev/null || true
echo "===== CUDA TOOLKIT =====" && nvcc --version 2>/dev/null || true
echo "===== COMPILER =====" && gcc --version 2>/dev/null | head -n 1
echo "===== PYTHON =====" && which python && python --version
echo "===== CONDA =====" && conda info --envs 2>/dev/null || true
echo "===== PYTORCH =====" && python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('cuDNN:', torch.backends.cudnn.version())"
echo "===== INSTALLED PACKAGES =====" && python -m pip freeze
-->
