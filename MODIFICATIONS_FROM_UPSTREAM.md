# AOFS 相对作者源代码的完整变更台账

上游仓库：`https://github.com/lalalayong/AOFS`

比较基线：`289af16d3dbbdfd1c0f98fb0dddab350be77493a`

本任务开始日期：2026-07-17

## 1. 使用与维护规则

本文档是本仓库相对上述作者基线的唯一总变更台账。范围包括所有源代码、配置、脚本、测试和文档；每次相关文件发生变化，必须在同一个修改批次更新本文档。

每条记录包含路径、上游行为、当前行为、原因、影响范围和验证状态。标记含义：

- `历史差异`：2026-07-17 本任务开始前，工作区已经存在且未提交的修改；本任务不声明其作者身份。
- `本任务`：为 paper/robust 双配置复现工作新增或修改的内容。
- `生成物`：编译、缓存、权重或训练输出，不属于作者源代码，不逐个二进制文件展开。

## 2. 本任务开始前已存在的本地差异

以下记录来自任务开始时的 `git status`、逐文件 `git diff` 和未跟踪文件检查。

| 状态 | 路径 | 相对上游的行为变化 | 原因/影响 | 验证状态 |
|---|---|---|---|---|
| 历史差异 | `cfg/fewtunev5_nwpu_10shot.data` | `max_epoch` 从 50000 改为 10000；train/valid 从仓库相对路径改为 `/workspace/dataset/...` | 将约 500 epoch 缩短为约 100 epoch，并绑定服务器路径；同时影响 paper/robust 的历史训练 | 已核对 diff；属于此前效果偏低的重要原因 |
| 历史差异 | `cfg/fewyolov5_nwpu.data` | train/valid 改为 `/workspace/dataset/...` | 服务器路径适配，但不可移植 | 已核对 diff |
| 历史差异 | `data/nwpu_poly.yaml` | 数据根目录从 `./dataset` 改为 `/workspace/dataset` | 服务器路径适配，但不可移植 | 已核对 diff |
| 历史差异 | `dataset.py` | 取消 `scratch` 到 `tmp_scratch/basilisk` 的路径替换；四处 `np.int` 改为 `int`；移除 `load_metadict` 中无条件 `pdb.set_trace()` | 服务器/新版 NumPy 兼容，并避免训练无条件进入调试器 | 已核对 diff；尚无自动化回归测试 |
| 历史差异 | `image.py` | 两个溢出分支由 `pdb.set_trace()` 改为 `continue` | 避免异常样本使训练停在交互调试器 | 已核对 diff；可能静默跳过样本，需后续测试/告警 |
| 历史差异 | `models/common.py` | `np.float` 改为 `float` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `tools/gen_fewlist_nwpu.py` | shot 集合从 15–60 改为 3/5/10；数据根从 `C:\\dataset` 改为 `/workspace/dataset` | 适配论文 shot 和服务器，但仍为固定路径/固定抽样语义 | 已核对 diff |
| 历史差异 | `train.py` | `metaloader.next()` 改为 `next(metaloader)` 并在耗尽时重建迭代器；验证条件从 `if not noval or final_epoch` 改为 `if not noval` | 前者兼容新 DataLoader；后者导致 `noval=true` 时最终 epoch 也不验证，best checkpoint 失去指标依据 | 已核对 diff；后一个变化被确认是缺陷 |
| 历史差异 | `utils/datasets.py` | 三处 `np.int` 改为 `int`；polygon 标签拼接改为显式数值 class ID 与 float 坐标 | 新版 NumPy 兼容，并避免字符串/数值拼接造成标签 dtype 错误 | 已核对 diff；尚无自动化测试 |
| 历史差异 | `utils/general.py` | 两处 `np.int` 改为 `int` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `utils/loss.py` | feature map 边界张量在 clamp 上限处显式转为 `int` | 新版 PyTorch 参数类型兼容 | 已核对 diff |
| 历史差异 | `DOTA_devkit/ResultEnsembleNMS_multi_process.py` | 注释示例中的 `np.int` 改为 `int` | 文本兼容性清理，不改变运行行为 | 已核对 diff |
| 历史差异 | `DOTA_devkit/ResultMerge_multi_process.py` | 注释示例中的 `np.int` 改为 `int` | 文本兼容性清理，不改变运行行为 | 已核对 diff |
| 历史差异 | `DOTA_devkit/dota_evaluation_task2.py` | `np.bool` 改为 `bool` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `DOTA_devkit/dota_poly2rbox.py` | `np.float` 改为 `float` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `DOTA_devkit/hrsc2016_evaluation.py` | `np.bool` 改为 `bool` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `DOTA_devkit/mAOE_evaluation.py` | `np.bool` 改为 `bool` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `DOTA_devkit/ucasaod_evaluation.py` | `np.bool` 改为 `bool` | 新版 NumPy 兼容 | 已核对 diff |
| 历史差异 | `DOTA_devkit/polyiou.py` | 增加 SWIG 4.3.0 生成头和版本导入 | 重新生成 polygon IoU Python 包装器 | 已核对 diff；生成版本与上游不同 |
| 历史差异 | `DOTA_devkit/polyiou_wrap.cxx` | SWIG 4.1.1 生成结果替换为 SWIG 4.3.0 生成结果 | 新编译环境兼容；属于大型自动生成源码差异 | 已核对 diff 规模；应由接口文件重建验证 |
| 历史差异 | `utils/nms_rotated/__init__.py` | 相对导入改为 `utils.nms_rotated...` 绝对导入 | 改变扩展模块的导入解析 | 已核对 diff；需在包安装和仓库根运行两种方式测试 |
| 历史差异 | `utils/nms_rotated/nms_rotated_wrapper.py` | 扩展模块改为顶层 `import nms_rotated_ext`；`torch.arange` 显式使用检测张量所在 device | 适配本地扩展安装方式，并修复 CUDA/CPU 索引 device 不一致 | 已核对 diff；需 CUDA smoke test |
| 历史差异 | `utils/nms_rotated/src/poly_nms_cuda.cu` | 移除旧 THC API，改用 c10/CUDA stream、`cudaMalloc/cudaFree` 和 `C10_CUDA_CHECK` | 新版 PyTorch/CUDA 扩展兼容 | 已核对 diff；需在目标服务器重新编译并运行 |
| 历史差异 | `cfg/fewtunev5_nwpu_3shot.data` | 上游不存在；新增 3-shot、10000 max_epoch、`/workspace/dataset` 配置 | 历史 3-shot 服务器训练配置 | 已检查内容；未跟踪文件 |
| 历史差异 | `cfg/fewtunev5_nwpu_5shot.data` | 上游不存在；新增 5-shot、10000 max_epoch、`/workspace/dataset` 配置 | 历史 5-shot 服务器训练配置 | 已检查内容；未跟踪文件 |
| 历史差异 | `data/nwpu_traindict_bbox_3shot.txt` | 上游不存在；10 类映射到 `/workspace/dataset/nwpulist/box_3shot_*_train.txt` | 历史 3-shot support 字典 | 已检查内容；未跟踪文件 |
| 历史差异 | `data/nwpu_traindict_bbox_5shot.txt` | 上游不存在；10 类映射到 `/workspace/dataset/nwpulist/box_5shot_*_train.txt` | 历史 5-shot support 字典 | 已检查内容；未跟踪文件 |
| 历史差异 | `data/nwpu_traindict_bbox_10shot.txt` | 上游不存在；10 类映射到 `/workspace/dataset/nwpulist/box_10shot_*_train.txt` | 历史 10-shot support 字典 | 已检查内容；未跟踪文件 |
| 历史差异 | `data/nwpu_traindict_full.txt` | 上游不存在；10 类映射到 `/workspace/dataset/nwpulist/*_training.txt` | 历史全量训练字典 | 已检查内容；未跟踪文件 |
| 历史差异 | `models/lsknet.py` | 上游不存在；新增两个值为 `None` 的 LSKNet 占位符 | 满足某处导入，但不提供可用模型 | 已检查内容；未跟踪文件，后续需验证是否仍必要 |
| 历史差异 | `tools/convert_dota_to_labels.py` | 上游不存在；把 DOTA polygon 标注通过 `minAreaRect` 转为五列旋转框标签，并跳过 difficult=2 | 历史数据转换工具；其输出格式需要与实际 loader 再核对 | 已检查内容；未跟踪文件 |
| 历史差异 | `results_comparison.md` | 上游不存在；新增历史结果对比报告 | 训练报告，不影响代码 | 已检查存在；内容编码显示异常 |
| 历史差异 | `results_summary.md` | 上游不存在；新增历史结果摘要 | 训练报告，不影响代码 | 已检查存在；内容编码显示异常 |
| 历史差异 | `saved_weights/results_summary.md` | 上游不存在；权重目录内新增历史结果摘要 | 训练报告，不影响代码 | 已检查存在；内容编码显示异常 |

## 3. 本任务引入的差异

| 状态 | 路径 | 相对上游的行为变化 | 原因/影响 | 验证状态 |
|---|---|---|---|---|
| 本任务 | `docs/superpowers/specs/2026-07-17-aofs-dual-profile-reproduction-design.md` | 上游不存在；新增单仓库 paper/robust 双配置设计、评估口径、测试和验收标准 | 在改代码前冻结范围，防止复现目标与稳健改进混在一起 | 已完成书面自检；等待用户审核 |
| 本任务 | `docs/superpowers/specs/2026-07-18-validation-label-count-design.md` | 上游不存在；新增零 True Positive 时仍正确统计真实标签的独立修复设计、测试和服务器同步边界 | 固化 CUDA 冒烟测试暴露的验证汇总根因，避免把数据问题与指标显示问题混淆 | 已完成书面自检；用户已于 2026-07-18 审核通过 |
| 本任务 | `docs/superpowers/plans/2026-07-17-aofs-dual-profile-implementation.md` | 上游不存在；新增九阶段、测试先行的双配置实施计划 | 把已通过的设计映射到具体文件、失败测试、实现接口和验证命令 | 已完成计划自检；按 inline execution 执行 |
| 本任务 | `docs/superpowers/plans/2026-07-18-validation-label-count-implementation.md` | 上游不存在；新增验证标签计数修复的 RED/GREEN、完整回归、GitHub 发布和服务器冒烟验收计划 | 将最小代码修复与服务器同步过程拆成可审计步骤，避免直接手改服务器造成分叉 | 已完成计划自检；等待用户选择执行方式 |
| 本任务 | `.gitignore` | 上游不存在；新增 Python 缓存、原生扩展构建目录、数据缓存、训练输出和权重目录忽略规则 | 防止可再生成文件再次污染源代码差异清单；不删除或忽略源代码 | 已核对规则；未忽略 `*.py`、配置、脚本、测试或文档 |
| 本任务 | `aofs/__init__.py` | 上游不存在；新增 AOFS 复现实验辅助包入口 | 为可测试的配置、抽样、评估和汇总逻辑提供稳定命名空间 | `python -m unittest tests.test_experiment -v` 的 RED 阶段已确认缺少包时失败 |
| 本任务 | `aofs/experiment.py` | 上游不存在；新增 `.data`/dataset YAML 路径覆盖与所有字符串环境变量解析、训练 epoch 换算、验证时机、实验身份和运行清单辅助函数 | 把会影响复现的决策从 GPU 训练循环中分离；支持 profile 中的 `${AOFS_SEED}` 以及路径变量 | dataset YAML root 和非路径 seed 展开测试均先 RED；`python -m unittest tests.test_experiment -v`：8 项通过 |
| 本任务 | `aofs/datasets.py` | 上游不存在；新增 NWPU/DIOR 单一类别顺序及 novel/base 注册表 | 消除评估脚本把 NWPU 预测按 DIOR 类别解释的错误 | `python -m unittest tests.test_obb_eval -v`：5 项通过 |
| 本任务 | `aofs/obb_eval.py` | 上游不存在；新增预测 JSON 转 DOTA Task1、逐类 AP、novel/base/all OBB mAP@0.5、逐类预测数和排除 difficult 的 GT 正样本数 | 缺失预测类别固定按 AP=0 计入分母，并输出可审计的机器可读指标 | AP/类别测试通过；预测/GT 计数契约先 RED 后 GREEN；OBB 测试最终 7 项通过 |
| 本任务 | `aofs/fewshot.py` | 上游不存在；新增 DOTA/NWPU 候选读取、精确 K 实例选择、可选单图上限以及 seed 专属 support/meta/audit 输出；兼容根/子集 `labels` 和作者标准 `training/labelTxt` | paper 保留无单图上限；robust 可限制同图贡献；标准 README 数据布局无需额外复制标签即可抽样 | 5 项初始测试通过；切换合成数据到 `training/labelTxt` 后先 RED 再 GREEN |
| 本任务 | `DOTA_devkit/dota_evaluation_task1.py` | 作者入口把 DIOR 全类别与 NWPU novel/base 混用，并跳过无预测类别缩小分母；现在强制 `--dataset`、调用共享 OBB 评估器，把 NumPy/polyiou 延迟到实际评估，并以 `from DOTA_devkit import polyiou` 同时支持脚本和训练内模块调用 | 修正 NWPU 类别 ID、mAP 分母及训练内周期评估找不到包内 `_polyiou` 的问题；`--help` 不依赖扩展 | 类别/CLI 5 项通过；package-aware 导入契约另行先 RED 后 GREEN；真实 polygon IoU 留待服务器 |
| 本任务 | `tools/gen_fewlist_nwpu.py` | 作者/历史版本使用源码内固定路径、固定 seed 2018 和全局输出目录；现在改为 `--data-root/--output-root/--profile/--shot/--seed` CLI，并调用 seed 专属 split 生成器 | 消除服务器路径硬编码和不同 seed 相互覆盖；paper 默认无单图上限，robust 默认每图 1 个实例 | CLI help RED 已确认旧脚本依赖 NumPy且缺少新参数；GREEN 及独立 help 命令通过 |
| 本任务 | `cfg.py` | 作者代码从 meta 文件名推断 shot 且不记录 profile/seed/support 根；现在优先读取显式 `shot` 并保存 profile、seed 和 seed 专属 support 根 | 避免动态输出文件名改变 shot 推断，并隔离不同实验身份 | `python -m unittest tests.test_support_paths tests.test_experiment -v`：10 项通过 |
| 本任务 | `dataset.py` | 在历史兼容修改基础上，tuning support 标签现在可从 `cfg.support_root/<class>/<image>.txt` 读取，未设置时保留作者路径替换 | 允许 paper/robust 和不同 seed 使用互不覆盖的 support 标签 | support 路由 RED 后，组合测试 10 项通过 |
| 本任务 | `train.py` | 在历史 DataLoader 修改基础上，新增 profile/stage/dataset/shot/seed/data-root/meta/support-root 与 OBB 验证参数；dataset YAML root 可覆盖；恢复周期/最终验证；按 novel OBB mAP@0.5 或 HBB fitness 选择 `best.pt` 并同步 `best_obb_metrics.json`；checkpoint/manifest 写入实验身份；仅同身份 `--resume` 恢复训练状态；仅 legacy 继续在 `increment_path` 后追加 `_tuning` | 修复最终不验证、验证 loader 缺失、二阶段错误继承状态和论文指标无法选优；新 profile 的命名目录不再因后追加 `_tuning` 绕过自动递增而覆盖 | 主契约 8 项、best 指标及目录后缀契约均先 RED 后 GREEN；语法编译通过，真实 CUDA/OBB 流程留待服务器 |
| 本任务 | `val.py` | 作者代码只在存在预测时写 JSON，训练内调用会生成不稳定的空 stem 文件，且独立验证无法解析可移植 profile；现在支持 `prediction_stem`、空预测写 `[]`、非空 COCO 才用 pycocotools，并为独立验证解析环境路径/覆盖 dataset root | 保证每轮 OBB 文件命名可预测、无预测类别正确计零，同时让训练与独立评估共用服务器无关配置 | 两轮 RED 分别确认输出契约和独立验证路径缺口；`python -m unittest tests.test_val_contract -v`：3 项通过 |
| 本任务 | `tests/__init__.py` | 上游不存在；新增标准库 unittest 测试包标记 | 允许不安装 pytest 时运行回归测试 | 已由 unittest 成功发现测试模块 |
| 本任务 | `tests/test_experiment.py` | 上游不存在；新增 `.data` 路径/非路径环境变量、dataset YAML root、epoch、`noval` 最终验证、周期验证和 resume 身份测试 | 防止硬编码路径、`${AOFS_SEED}` 未展开、100/500 epoch 混淆及最终不验证 | 初始包缺失、root helper 和 seed 展开均先 RED；GREEN 8 项通过 |
| 本任务 | `tests/test_obb_eval.py` | 上游不存在；新增 NWPU 映射、缺失类零 AP 分母、非空预测评估、逐类预测/正样本数、CLI 无重依赖及 package-aware polyiou 导入测试 | 固化论文 novel OBB 口径和报告审计字段，避免脚本/训练内扩展导入差异 | 模块、顶层依赖、package 导入和计数分别先 RED；GREEN 7 项通过 |
| 本任务 | `tests/test_fewshot.py` | 上游不存在；新增 exact-shot、同 seed 可复现、单图上限、数据不足、标准 `training/labelTxt` 完整 split 产物和 CLI 参数测试 | 防止 support 集中于单图、保证抽样可审计并覆盖作者数据目录约定 | 模块/CLI 和 `labelTxt` 路由分别先 RED；GREEN 5 项通过 |
| 本任务 | `tests/test_support_paths.py` | 上游不存在；新增 seed support 路径、cfg 显式 shot/support、MetaDataset 路由和 train CLI 合约测试 | 保证不同 seed 不再复用同名 `labels_1c` | RED 已按预期因缺少 helper/路由/参数而失败；GREEN 4 项通过，组合测试共 10 项通过 |
| 本任务 | `tests/test_val_contract.py` | 上游不存在；新增 val CLI/函数 prediction stem、空 JSON 保存以及独立评估 profile 路径解析合约测试 | 防止无检测轮次缺结果、epoch JSON 覆盖及服务器环境变量路径无法用于独立验证 | 输出与路径集成均先 RED；GREEN 3 项通过 |
| 本任务 | `tests/test_train_contract.py` | 上游不存在；新增周期验证、活动验证 loader、OBB metric/最佳指标文件、checkpoint 身份、manifest/resume、二阶段状态隔离、dataset root、命名目录和 CLI 参数源码契约 | 在无本机 CUDA 环境下固定训练主流程必须具备的关键集成点 | 8 项主契约、`best_obb_metrics.json` 及命名目录契约均先 RED 后 GREEN |
| 本任务 | `cfg/paper/nwpu_base.data` | 上游不存在；新增以 `${AOFS_DATA_ROOT}`、`${AOFS_BASE_META}`、`${AOFS_SEED}` 表达的 NWPU paper base-stage 配置 | 为后续服务器 base 训练提供不绑定机器路径的显式身份；full-meta 文件仍需按操作指南生成/指定 | profile 测试覆盖环境变量约定；真实 base 训练留待服务器 |
| 本任务 | `cfg/paper/nwpu_3shot.data`、`cfg/paper/nwpu_5shot.data`、`cfg/paper/nwpu_10shot.data` | 上游不存在；新增 paper 3/5/10-shot 配置，显式 `shot/profile/seed/support_root`，并恢复 `max_epoch=50000, repeat=100`（500 epoch） | 恢复论文训练预算，隔离每个 seed 的 support/meta 路径 | profile 文件缺失阶段 5 项测试按预期报错；GREEN 5 项通过 |
| 本任务 | `cfg/robust/nwpu_3shot.data`、`cfg/robust/nwpu_5shot.data`、`cfg/robust/nwpu_10shot.data` | 上游不存在；新增 robust 3/5/10-shot 配置，与 paper 使用相同训练预算/路径结构但身份独立 | 只把 robust 差异限定在 scene-aware support 抽样，便于与 paper 公平对比 | 同上；GREEN 5 项通过 |
| 本任务 | `cfg/paper/hyp.finetune_nwpu.yaml` | 上游不存在；从作者 NWPU 配置派生并把 momentum 从代码文件中的 0.937 对齐为论文报告的 0.999；保留 `lr0=0.001, weight_decay=0.0005` 及其余增广/损失项 | 消除历史运行超参数与论文文字口径的偏差 | profile 测试精确检查三项论文超参数并通过 |
| 本任务 | `cfg/robust/hyp.finetune_nwpu.yaml` | 上游不存在；与 paper hyperparameter 内容一致，仅注释/路径标明 robust 身份 | 让首轮 robust 对比只改变抽样策略，不混入 optimizer 或增广变化 | profile 测试精确检查三项论文超参数并通过 |
| 本任务 | `scripts/train_nwpu_paper.sh` | 上游不存在；新增严格 Bash 启动器，生成 paper seed split，以隔离 project、OBB novel metric、完整身份和 `--patience 500` 启动；OBB `{:s}` 默认模板用显式分支赋值 | 固化服务器复现命令，防止提前截断 500 epoch，并避免嵌套 parameter expansion 吃掉模板右花括号 | patience 和模板契约分别先 RED；Git Bash 语法及 xtrace 默认值检查通过 |
| 本任务 | `scripts/train_nwpu_robust.sh` | 上游不存在；新增同预算 robust 启动器，split 默认每图 1 实例，隔离输出、`--patience 500` 和安全 OBB 模板赋值 | 降低 support 场景集中风险，同时保持 paper 完整预算和正确评估路径 | 同上；Bash 语法检查退出码 0 |
| 本任务 | `scripts/eval_nwpu_obb.sh` | 上游不存在；新增确定性 split 重建、独立 JSON 导出和共享 NWPU polygon AP 串联；安全生成含 `{:s}` 的默认 annopath | 使用 fixed-denominator novel/base/all OBB mAP@0.5，并避免 Bash 将模板误解析成 `{:s.txt}` | 模板测试先 RED 后 GREEN；Bash 语法检查退出码 0；真实 GPU/扩展留待服务器 |
| 本任务 | `tests/test_profiles.py` | 上游不存在；新增 profile 身份、500-epoch/禁止提前停止、论文超参数、隔离输出、共享评估及 Bash `{:s}` 模板安全测试；轻量解析避免 PyTorch/PyYAML | 在本机验证服务器配置与 shell 值语义 | 文件缺失、patience 和嵌套模板分别 RED；最终 5 项全 GREEN |
| 本任务 | `aofs/results.py` | 上游不存在；新增 OBB 指标与父级 run manifest 严格配对、三 seed 均值/样本标准差汇总及 profile/dataset/shot/stage 混用拒绝 | 让论文 mean ± std 可复算并防止把不同实验组错误合并 | `aofs.results` 缺失阶段测试按预期导入失败；GREEN 3 项通过 |
| 本任务 | `tools/summarize_obb_runs.py` | 上游不存在；新增接收三个以上 `obb_metrics.json` 和 `--output` 的无重依赖 CLI | 在服务器把每个 run 的最佳 novel OBB 指标汇总为机器可读 JSON | `python tools/summarize_obb_runs.py --help` 退出码 0并列出 metrics/`--output` |
| 本任务 | `tests/test_results.py` | 上游不存在；新增三 seed 样本标准差、父目录 manifest 发现和混合 profile 拒绝测试 | 固化论文重复实验统计口径与输入身份约束 | RED 为缺失模块；`python -m unittest tests.test_results -v`：3 项通过 |
| 本任务 | `docs/AOFS_REPRODUCTION.md` | 上游不存在；新增同步边界、服务器扩展重建、数据预检、base/paper/robust 训练、独立 OBB 复评、三 seed 汇总、论文目标和服务器验收顺序 | 给出从本地源码到服务器完整实验的唯一操作路径，并明确论文数值在实测前不保证 | 已逐项对照当前 CLI/profile/output 名称；命令级 GPU 验证留待服务器 |
| 本任务 | `util.py` | `read_data_cfg` 原来对所有 `=` 分割；现在只分割第一个 `=` | 允许配置值安全包含环境变量或带等号的路径值 | `python -m unittest tests.test_experiment -v`：8 项通过 |
| 本任务 | `MODIFICATIONS_FROM_UPSTREAM.md` | 上游不存在；新增本文档 | 满足所有相对作者源码差异可追踪的要求 | 已用任务开始时的 Git 状态和逐文件 diff 建立初始清单；后续每个补丁同步维护 |

## 4. 编译、缓存和训练生成物

这些路径在任务开始前已存在且未被上游跟踪。它们不是作者源代码修改，不作为实现补丁提交；同步服务器或清理前需由用户确认。

2026-07-17 经用户确认后，已清理所有 `__pycache__/`、`DOTA_devkit/build/`、`utils/nms_rotated/build/`、`utils/nms_rotated/nms_rotated.egg-info/` 和 `utils/nms_rotated/src/poly_nms_cuda.cu.bak`。根目录编译扩展、历史运行和权重继续保留；下表仍保留任务开始时的生成物审计记录。

| 状态 | 路径/类型 | 说明 |
|---|---|---|
| 生成物 | `DOTA_devkit/_polyiou.cpython-310-x86_64-linux-gnu.so` | Linux CPython 3.10 polygon IoU 编译扩展 |
| 生成物 | `DOTA_devkit/build/` | polygon IoU 编译目录 |
| 生成物 | `utils/nms_rotated/build/` | rotated NMS 编译目录 |
| 生成物 | `utils/nms_rotated/nms_rotated.egg-info/` | 本地安装元数据 |
| 生成物 | `utils/nms_rotated/src/poly_nms_cuda.cu.bak` | CUDA 源码备份，不属于上游文件 |
| 生成物 | 根目录及各子目录的 `__pycache__/`、`*.pyc` | Python 字节码缓存 |
| 生成物 | `runs/` | 历史训练和验证输出 |
| 生成物 | `saved_weights/`（除上节单列的 Markdown） | 历史保存权重与结果文件 |
| 生成物 | `weights/` | 模型权重文件 |

## 5. 完整性检查方法

每个实现批次结束时执行以下检查，并将新出现的源代码/配置/脚本/测试/文档路径补入第 3 节：

```powershell
git status --short
git diff --name-status 289af16d3dbbdfd1c0f98fb0dddab350be77493a
git ls-files --others --exclude-standard
```

最终交付时，第 2、3 节的路径并集应覆盖上述命令中除第 4 节生成物以外的全部路径。验证结果应记录具体命令和结论，不使用“应该通过”代替实际检查。

### 2026-07-17 最终本机验收记录

- `python -m unittest discover -s tests -v`：44 项测试全部通过，0 failure、0 error，退出码 0。
- `python -m py_compile aofs/__init__.py aofs/experiment.py aofs/datasets.py aofs/obb_eval.py aofs/fewshot.py aofs/results.py tools/gen_fewlist_nwpu.py tools/summarize_obb_runs.py DOTA_devkit/dota_evaluation_task1.py cfg.py dataset.py train.py val.py util.py`：退出码 0。
- `python tools/gen_fewlist_nwpu.py --help`、`python tools/summarize_obb_runs.py --help`、`python DOTA_devkit/dota_evaluation_task1.py --help`：均显示预期参数并以退出码 0 结束。
- `D:\github\Git\bin\bash.exe -n scripts/train_nwpu_paper.sh scripts/train_nwpu_robust.sh scripts/eval_nwpu_obb.sh`：退出码 0；另以 `PYTHON_BIN=echo` xtrace 验证默认 annopath 保持为 `/data/evaluation/labelTxt/{:s}.txt`。
- 差异路径自动比对：`git diff --name-only <基线>` 与 `git ls-files --others --exclude-standard` 中，仅 6 个已删除的历史 `__pycache__/*.pyc` 未逐文件出现在反引号路径中；它们由第 4 节的统一生成物规则覆盖。其余源代码、配置、脚本、测试和文档路径均在第 2 或第 3 节逐项记录。
- 本任务范围 `git diff --check -- train.py val.py dataset.py cfg.py util.py DOTA_devkit/dota_evaluation_task1.py tools/gen_fewlist_nwpu.py` 退出码 0；新增文件尾随空白扫描无匹配。全仓相对基线的 `git diff --check` 仍报告任务开始前已有的 SWIG 4.3.0 生成文件 `DOTA_devkit/polyiou_wrap.cxx` 和 CUDA 文件 `utils/nms_rotated/src/poly_nms_cuda.cu` 空白，本任务为保护历史用户修改未重写这些文件。
- 当前分支为 `aofs-dual-profile`，普通仓库（非额外 worktree）。Git `user.name`/`user.email` 未配置，因此未创建提交；所有已验证文件保留在当前工作区。
