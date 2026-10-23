# MAMMA 512 点 ROI 小样本适配（2026-10-13）

状态：五个有框槽位完成真实 Dataset 回读；一个无框槽位保留原因。全部产物 **integration_only**，未批准正式训练/验证来源。只用了原生 oracle 框与可见实例 oracle mask；框增强、旋转、极端裁剪和随机 mask 均关闭。未构造 MammaNet、执行 forward、训练或渲染。

## 模板和资产编号

`data/body_models/downsampled_verts/verts_512.pkl` 是 MAMMA 的 `[512,10475]` one-hot 顶点选择矩阵。`BEDLAM_WD.py` 在 `num_joints=512` 下按原行序取 `argmax`；`MammaNetDecoder` 的 512 queries 输出不做重排。`_smplx_512_body_parts.json` 覆盖 512 编号，部位归属不代替几何。此定义不是 R138 插值。

本地 standard 与 neutral locked-head 的既有审计（`densekp_retrain/outputs/audits/bedlam_r138_compatibility_a2_20261009/`）验证有序 faces、每个 face corner UV、所有 vertex group/权重逐项相同。该本地 locked-head 位于 `densekp_retrain/assets_external/smplx_blender_addon/data/smplx_model_lh_20230302.blend`，无需 MAMMA 自身目录另有副本。512 个选择顶点中 50 个在 locked-head 头部改型区发生坐标变化，但仍保留原编号、面、UV 和绑定；不能要求形状坐标相等。五个有框槽位的三份实际身体缓存的有序 faces 均与该 locked-head 标准面完全一致。M02 资产谱系记录为标准 male SMPL-X，M03/M04 由本地 neutral locked-head 生成；只将配置中这三份已核实缓存列入 `qualified_assets`。Dataset 对未知资产或有序面不一致拒绝输出，不能由顶点数自动放行。未验证鞋/脚趾改型、重拓扑、其它性别/发行模型、外部未知 posed cache；这些均 pending。

几何使用准确 `asset_frame_index` 的身体顶点，单次应用对应人物的 `scene_matrices[row,person]`，再用 `annotations.npz:camera_to_world_erp` 的 CV 相机矩阵投影到完整 ERP。逐项核对 `person_id`、源帧、整数 source tick/time base 和合成帧；不叠加第二次 root 位移。以同一链重算原 R138，与保存的世界点最大差 6.97e-8 m、ERP xy 最大差 0.000247 px。该回读只验证变换链，512 编号语义依靠上述模板、资产制作谱系和面/UV/权重证据。

## Dataset 合同

`landmarks/lib/datasets/mamma_roi_pilot.py` 读取审计 inventory 指定的固定观察槽。图像和局部人物 mask 直接从 ROI PNG 解码；生产读取从不回填 5760×2880 空画布。原框 `xyxy` 按部署 `1.2×` 扩张后适配 384:512，滤波 sigma 和 `ceil(4σ)+3` 支持矩形先做越界检查；越出 ROI 直接拒绝。局部采样仿射由完整 ERP 仿射平移而来，图像双线性、mask 最近邻；原二值 mask 的 0/255 转为模型 0/1。图像使用活动部署 `fast_inference=True` 的 OpenCV GaussianBlur 条件预滤波、BGR→RGB 和 ImageNet 均值/方差。点同时输出完整 ERP xy、384×512 模型像素 xy、有序顶点 ID、正逆仿射。训练若需要 `[-1,1]`，按原 BEDLAM 路径把模型 xy 分别除以 `[384,512]` 再 `*2-1`；本阶段保留原始模型像素，避免混淆。

输出将 `geometry_reliable`、完整校准 physical mask 的域内状态、ROI coverage、crop support 和 `coordinate_loss_valid` 分开。坐标有效性要求可靠投影、物理域内、ROI 已渲染、裁剪内；遮挡不自动取消可靠坐标。`visibility`、`contact`、`floor_contact` 均为 `-1/unknown`，各自 loss-valid 全 false；本批没有重解释原 loss 或训练这些头。局部 `valid_mask` 同仿射产生 `coverage_crop`；物理域依据单独校准 mask，未由 RGB 黑色推断。R138 在五例共 690 点中，校准 mask 采样与保存的 `in_view` 无不一致；这不生成 512 点遮挡证据。

## 定向 CPU 验收

参考路径在验证专用内存中把同一 ROI 回填为完整画布，动态加载**原仓库**未改的 `vitdet_dataset.py` 当前有效实现并调用 `ViTDetDataset(fast_inference=True)`。回填仅用于验证。生产路径仍只读局部 ROI。原生框的完整滤波支持都在 ROI 内。先核对原全图裁剪，再核对活动部署 `fast_inference=True` 路径；早期局部中心 float32 计算曾造成平移项最多 0.0029 px 偏差，改为先求完整 ERP 仿射再变换到局部后消失。最终五例图像、归一化图像、mask 和完整仿射均逐元素一致；图像与 mask 差异数/最大/平均均为 0。逆变换回 ERP 误差小于 1.2e-12 px。测试还拒绝非法 ROI 尺寸、越界支持，保留无框原因；未测扰动框。

| 槽位 | 相机/人物 | 坐标有效/512 | 状态 |
| --- | --- | ---: | --- |
| e01 rig_edge f0 cam2 p1 | 骑行 | 491 | 图/mask/仿射一致 |
| e01 rig_edge f0 cam4 p1 | 骑行 | 475 | 一致 |
| e02 rig_front f0 cam2 p2 | 行人 | 512 | 一致 |
| e02 rig_front f0 cam4 p2 | 行人 | — | `person_absent`: `outside_physical_domain_no_instance_box`，未生成训练框 |
| e03 rig_back f15 cam2 p2 | 行人 | 512 | 一致 |
| e03 rig_front f15 cam4 p1 | 骑行 | 228 | 一致，物理边缘样例 |

明细、来源、误差和五份小型 512 点 NPZ、模板资格 JSON 和两张诊断图见 `outputs/acceptance/mamma_data_adapter_20261013/`。CPU 数值验收含失败重试和 checkpoint 安全读取远低于 20 分钟；本轮未运行全库 discover、GPU 或小网络测试。原 checkpoint 的安全元数据读取在现有 `torch 2.0.0+cu118` 被 OmegaConf `ListConfig` 拒绝，因此 epoch/step、key/shape、checkpoint 内配置均未获取，**没有严格加载结论**。首次尝试的 `mmap` 参数也不被此 PyTorch 版本支持；没有关闭 `weights_only` 或安装新版本。

## 限制和下一批

当前 Dataset 仅接受这六个固定槽位、三个已列出身体缓存及原生 oracle 框。ROI 支持不代表扰动框、旋转或极端裁剪可用；新的矩形框须重新算含滤波 halo 的完整支持并有新 ROI。旧 compact 的深度/实例缓冲不足以恢复 512 点遮挡；`visibility/contact/floor-contact` 继续 unknown。原 `loss.py` 将第三值作为 log sigma（`exp(third)`），而 `run_ma_2d.py` 和下游部分代码按 log variance 用 `sqrt(exp(third))`；单独记录，不修改 ma_3d。

下一批先固定同一五有框槽的原权重 PyTorch 基线输入/后端，解决受控安全读取并做模型 key/shape **严格加载**，再核对当前下游第三值的实际消费；同时记录 oracle 与预测框/mask 的口径。若要训练 visibility，需要生成端在 live 深度/实例缓冲中追加点级证据并回读。来源许可、KIT 演员独立性、预训练暴露和动态观感尚未结案；同场景多 rig、邻帧、衍生衣发不能拆成独立 train/val。训练层、步数、预算和框增强 off/on 尚未冻结。
