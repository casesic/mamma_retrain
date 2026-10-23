# MAMMA 原权重五样本 PyTorch 定位基线

日期：2026-10-15。状态：**严格加载与固定五槽推理流程通过**。这是五个合成样本的原权重 XY 定位记录，全部为 `integration_only`；没有性能通过门槛、泛化结论、微调收益声明或新增训练。新实现和产物仅在 `mamma_retrain`。查看页：`outputs/experiments/mamma_original_baseline_pilot_20261015/index.html`。

## 固定输入和执行边界

沿用 2026-10-13 已验收的 `MammaROIPilot`：e01 双路骑行、e02 正面行人、e03 背面行人和正面骑行，共五个有框槽。e02 的 f0 cam4 p2 仍为 `person_absent / outside_physical_domain_no_instance_box`，未造框或前向。所有输入是原生 oracle 框、可见实例 oracle mask、原 ROI，活动部署 `fast_inference=True` 的 512×384 RGB/mask 裁剪；框增强、检测器、随机 mask、旋转和极端裁剪均未运行。每次 batch1、float32、eval、`torch.inference_mode()`；单 GPU0，禁用 AMP、TF32 和 cuDNN benchmark，种子 0。五例各一次前向，再重建模型、严格重载同一 tensor state，对预设 slot0 一次复核前向；正常路径合计六次，没有 warmup/反向/optimizer/参数更新。

## 权重来源、受控读取和严格加载

活动 PyTorch 部署 `/home/yy/Repos/riding-data/run_model_comparison.py:203-205` 从 `/home/yy/Repos/mamma/data/weights/ma_2d/mamma_mask_full_cvpr.ckpt` 读取 `state_dict` 并 `strict=True` 加载。原 MAMMA `inference/assets.py` 把该路径登记为官方 MAMMA `weights/mamma_mask_full_cvpr.ckpt` 资产；当前本地文件 1,621,536,317 字节、存在于原仓库 HEAD `588492f...` 的工作环境中，**权重文件自身未被 Git 跟踪**，没有独立下载日志或字节级来源证明。用户指定此现有部署权重，本批仅对此固定文件作受控读取，不下载资产或读取凭据。ViTPose 初始化文件缺失，TensorRT engine 是另一路部署二进制，均未当作训练 checkpoint。

现有 torch `2.0.0+cu118` 的 `weights_only=True` 不能解出 checkpoint 内 OmegaConf `ListConfig`，这是格式限制，不是权重损坏。按活动部署方式，在只访问该固定路径的 CPU 子进程调用一次 `torch.load(weights_only=False,map_location='cpu')`；子进程限制外溢副作用，**不消除 pickle 执行风险**。导出原 `state_dict` 的 282 个 CPU tensor 到任务目录 `tensor_state.pt`（540,487,469 字节），没有复制 optimizer、callbacks 等对象，原 checkpoint 未改。随后主进程两次以 `weights_only=True` 安全读取纯 tensor 文件。checkpoint 报 `epoch=96`、`global_step=500000`、Lightning `2.4.0`；不含 `hyper_parameters`，因此 checkpoint 内训练配置 unavailable。原文件及新 tensor state 均未计算 SHA256。

部署 YAML 构造 `DenseLdmks2DViT`：ViT 768 宽、12 层，512 个 MammaNet query，mask encoder，坐标/visibility/contact/floor-contact 四头。模型与权重均为 **282 key**，有序名称逐项对应；missing 0、unexpected 0、shape mismatch 0、前缀转换无，`strict=True` 成功。backbone 149 key 在权重中完整存在，缺失的 ViTPose 初始化会被完整 checkpoint 严格覆盖。二次重构严格加载同样成功。无法对照 checkpoint 内超参数，只能证明与当前活动部署构造实现的 key/shape 和实际前向对应。

实际 `lib.models`、ROI adapter 和基线模块均从新副本 import。`PYTHON_DOTENV_DISABLED=1` 阻止上游 import 读取 `.env`；没有修改原代码或共享环境。设备为 GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`（RTX 3090）；GPU1 的 Blender 进程未触碰。峰值 torch GPU allocated 652,599,808 字节。原 checkpoint CPU 导出 7.74 秒，主严格加载与六次前向 6.46 秒；显示修正只做 CPU 读取，无新增前向。全部数值工作远低于 20 分钟。

## 坐标与结果

配置 `data.train.normalize_plus_min_one=true`，网络 `joints2d[:,:,:2]` 为 `[-1,1]` 模型坐标，第三值原样保留。依活动部署公式 `u=(raw_xy+1)/2; ERP=u*box_size-box_size/2+box_center` 还原；独立把 `u*[384,512]` 经上一批 ERP 仿射逆变换核对。五例两路径最大差为 **0.000139 ERP px**，是原框/仿射 float32 锚点与 double 运算的数值差，未按误差选择路径；评价固定采用活动部署框公式。预测落裁剪外仍计误差。四个输出形状分别为 `[1,512,3]`、三个 `[1,512,1]`，输入 torch.float32、输出 float32，所有头非有限值计数均 0。

坐标评价沿用可靠几何∧物理域内∧ROI coverage∧模型裁剪内；没有按 visibility 或第三值筛点/加权。欧氏误差单位为完整 5760×2880 ERP 像素，不与 DenseKP 256-square 指标比较。

| 样本 | 有效点/512 | ERP 均值 px | 中位数 px | P90 px |
| --- | ---: | ---: | ---: | ---: |
| e01 rig_edge f0 cam2 p1 | 491 | 20.913 | 9.207 | 81.708 |
| e01 rig_edge f0 cam4 p1 | 475 | 11.102 | 7.879 | 25.409 |
| e02 rig_front f0 cam2 p2 | 512 | 3.159 | 1.338 | 9.252 |
| e03 rig_back f15 cam2 p2 | 512 | 3.624 | 2.746 | 7.336 |
| e03 rig_front f15 cam4 p1 | 228 | 3.768 | 1.594 | 10.592 |

合计 **2,218 个有效点**；点加权均值 **8.960 px**，五样本等权均值 **8.513 px**。e01 同时刻同人物双路共有 **454 个有效编号**；在此固定交集内，cam2 各点对自身 GT 的均值 20.833 px、cam4 为 11.462 px，逐编号两路自身残差平均的均值 16.148 px。这是**配对 GT 残差**，不相减两台相机的像素坐标，也不代表三角化或跨视角合图误差。e01 cam2 的 P90 较大；本批不据此改样本、框或权重。

再次重建、严格加载后，slot0 的四个头与第一次输出逐元素一致：各头最大/平均绝对差均为 **0**。逐点 NPZ 含 512 编号、原始预测 xy/第三值、模型与完整 ERP xy、GT、坐标有效标记、四头原始输出及误差；`metrics.json` 含形状、dtype、分母、配对和设备记录。五张叠图基于实际 384×512 模型输入，绿色 GT、品红预测、灰色无效 GT，蓝色淡染代表物理无效区；展示每四个编号中的一个，评价仍用全部有效点。可见性 GT 统一标 `UNKNOWN`，ROI offset 写在图上。不生成虚构全景背景。

## 第三值与限制

训练 `landmarks/lib/models/models_2d/loss.py:48-62` 对第三值取 `sigma=exp(third)`，对应 log sigma；原 `run_ma_2d.py:252` 绘图使用 `sqrt(exp(third))`，对应 log variance。`optimization/utils/triangulation_functions.py:37-38` 根据 `channel_is_logvar` 选择 `exp(0.5*third)` 或直接缩放第三值；当前 `optimization.py:470,610,624` 的初始三角化调用默认 `use_uncertainties=False`、`channel_is_logvar=False`。本批保存原始第三值，**不把它当成未经校准的置信度**，不参与筛选或误差权重；不修改 ma_3d。visibility/contact/floor-contact 没有可信 512 GT，故不报这些头的准确率或遮挡分组精度。

五例仅证明加载与推理链可运行，并记录此固定输入下的定位表现。所有样本仍是 integration_only；来源许可、KIT 演员独立性、预训练暴露和动态观感限制沿用上一批。同场景多 rig、邻帧和衣发衍生不能拆成独立训练/验证。

下一批最小动作应依据此固定基线选择：先确认合格来源后扩大样本，或补一个有 ROI 支持证明的矩形框挑战，或准备已定预算的有界微调。此处不预定训练层与步数，不自动执行。
