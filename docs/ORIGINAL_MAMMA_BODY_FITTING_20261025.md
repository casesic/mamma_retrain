# 原 MAMMA 后接人体 fitting：真实双机合图对照

这次只比较两条链：**未微调 MAMMA，阈值 0.5 → 原演示合图**，以及**同一份 MAMMA 点位 → SMPL-X 人体 fitting → 同一演示合图**。两条链使用固定的 40 对同步 ERP、原人物 mask、20260320／20260311 各自的有效域、同一 D 人物配对和同一局部网格／E 归属／硬覆盖参数。第二条链只改变进入网格的点坐标，没有增加新可见点、改配对或重做检测。这里的人体 fitting 优化姿态、身体形状与两路相机人体参数；它不是合图内部的二维网格拟合。

原模型来源是 `outputs/experiments/mamma_original_baseline_pilot_20261015/tensor_state.pt`（既有记录的 SHA-256：`b69d98d4c4c9c3a63e88cfe83e32509f1f3cd28d9aa9e68413b30b2663aac1de`）。本次直接读取先前已用该模型生成的 `real_riding_manual_review_20261024/predictions_original.npz`，逐项核对 319 个目标 ID、模板顺序、仿射尺寸与有限坐标／概率；没有再运行 MAMMA 推理。0.5 只用于两侧原预测的有效点筛选，不改变保存的概率。

## 512 点如何接入原求解器

pano-stitch 的交付 C++ worker 原本只接 DenseKP 的 138 行观测，由 204 个 SMPL-X 顶点组成前向模型。MAMMA 的 512 行是**固定顺序的 SMPL-X 顶点 ID**，不能拿前 138 行或把行号当 DenseKP ID。本项目的[fixture 构建器](../scripts/fitting512/prepare_worker.py)先核对交付 fixture 的 204 行与只读 `SMPLX_NEUTRAL.npz` 的身体基形、形状方向、姿态方向和蒙皮权重一致，再按保存的 `ordered_vertex_ids` 选出 512 行。512×512 单位观测算子使第 *i* 个输入严格对应模板的第 *i* 个顶点。保留原基关节、10 维形状、手部、姿态／形状先验和分阶段 LM 数值代码。

worker 在本项目独立构建目录编译：只把原 C++ worker 的两处 138 行请求上界改为 512；原有求解器 C++ 源仍由只读 pano-stitch 路径编译，不修改外部项目或交付二进制。内部文件名 `densekp_regressor_138x204.f32` 是原 worker 的固定读取名，**新 fixture 的实际内容是 512×512 单位矩阵**，由元数据和测试核对，不是旧 DenseKP 回归器。项目适配位于[scripts/fitting512](../scripts/fitting512/)，实际执行文件是 `outputs/experiments/original_mamma_fitting_20261025/adapter/build/mamma_512_worker`；HMR2 冷启动权重为只读 pano-stitch 中的 `artifacts/human_fitting_delivery/hmr2/checkpoints/epoch=35-step=1000000.ckpt`。这两个资源都没有替换生产 worker 或部署模型。

每个有 D 配对的人物在当前帧独立做一次 HMR2 冷启动，视角按 pano-stitch 现有的边缘截断、有效点数、mask 框面积规则选定。框来自同一冻结人物 mask，HMR2 不重新检测。其 SMPL 姿态转 SMPL-X 后，用原求解链的球面方向规则设置初始平移；随后输入**双方共同有效的原 MAMMA 模板点**。完整 ERP 到 ERP_ROTATE 使用球面旋转，拟合结果按逆旋转回到原 ERP。跨接缝数值往返检查小于 0.002 像素。逐帧独立启动，不把这些相隔很远的图片当连续跟踪，也不借用其他人物状态。

实际优化保留交付 fixture 的配置：Stage 1 最多 20 个接受步，pose 最多 5 个，pose＋shape 最多 1 个，10 维 beta、固定 Charbonnier 与姿态／形状先验。`MAX_ACCEPTED_STEPS` 表示达到这组既定步数上限，不能把它称作已满足独立收敛阈值。拟合输出如果非有限或执行失败，清单记下原因并用原点对回退；网格门槛仍对拟合后的双侧点做物理支持和变形准入检查。没有配对的帧保留原链基础合图。

## 小样本贯通

按冻结清单顺序，先选 data3 的 `01145` 和 data4 的 `00518`，因为它们分别是两套标定中首个有原模型 D 配对的帧；这两个输入在查看 fitting 结果前已固定。前端配对身份和共同点数与原合图清单一致，分别为 44、117 点。两帧的 512 点 worker 都实际执行，状态为 `MAX_ACCEPTED_STEPS`；目标值分别从 895.70 降到 42.68、从 759.65 降到 8.12。拟合后点位移动中位数分别为 16.21、4.47 原 ERP 像素，原网格门槛均准入并产生不同于直接合图的图片。目标值与图像差异只证明链路工作，不证明真实点位更准或画面更好。

## 运行与看图

以下命令是**复现入口**，已有结果不应覆盖。运行前需设置新的独立输出目录，外部 pano-stitch 源和资产保持只读可访问。HMR2 使用现有 `camerahmr` Python 环境，C++ fitting 使用本项目构建的 CPU worker，真实合图沿用现有只读 Repos 模块：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
PY=/home/yy/conda_envs/camerahmr/bin/python
"$PY" scripts/fitting512/prepare_worker.py --output outputs/experiments/<新实验>/adapter --build
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mamma_fitting_mpl "$PY" scripts/fitting512/compare.py \
  --adapter outputs/experiments/<新实验>/adapter \
  --output outputs/experiments/<新实验>/comparison --limit-pairs 40
```

现有本轮结果在[对照清单](../outputs/experiments/original_mamma_fitting_20261025/comparison_gpu_trial2/index.json)。两个准备阶段失败现场另外保留：一次是沙箱 GPU 不可见，一次是 HMR2 视角对象属性名错误，均未得到有效 fitting；它们不混入此清单。旧原 MAMMA 合图继续从原清单只读引用，没有重写。这里的局部网格是 riding-data 的演示链，不是 canonical CPW 或最终公司合图算法。

## 完整 40 对结果

固定的 data3、data4 各 20 对均有原图、直接合图、fitting 合图及逐帧状态。原前端在 37 对帧中找到人物配对，另 3 对（data3/01529、data3/03816、data4/04708）没有配对，两栏都显示同帧基础拼接。37 对里共有 44 组跨机人物配对：30 帧各一组，7 帧各两组。44 组均通过 512 点 worker 得到有限输出，并被原网格门槛准入；最终为 **37 帧 WARPED、3 帧 NO MATCH**，与直接合图的帧级状态相同。没有把无配对帧从对照里删掉，也没有增加可见点或重配人物。

44 组输入共 3,460 个双方共同有效的模板点，中位数为每组 74 点。所有求解状态都是 `MAX_ACCEPTED_STEPS`，表示跑到预设步数上限；不是独立的收敛证明。各组目标值均下降，拟合投影范围相对原点范围的比值为 0.946–1.004，没有出现整体尺度塌缩；这些仍只是过程诊断。拟合后相对原点位的逐组位移中位数的中位数为 4.20 ERP 像素，最大单点位移 126.34 像素。HMR2 初始化累计 7.58 秒，LM 拟合累计 517.77 秒；两阶段合计约 525 秒，尚不包含读图、配对和合图。首两帧贯通约 44 秒，其后续批次约 739 秒，整批约 **13 分钟**，使用 GPU0（`GPU-d481acc7-106f-29e5-1298-8c0d398261b2`）做 HMR2，C++ worker 用 CPU。直接合图是已保存的原分支，本轮没有重算原 MAMMA 预测。

在原尺寸 5760×2880 的 PNG 上可以查看实际画面。初看 data3/01145 与 data4/00518 的快速预览，整体构图相近；局部像素确实发生变化，尚不能据此判断哪边更好。真实图片没有人工 512 点 GT，优化目标下降、网格准入或像素差都不是真实准确率。请在同一局部放大检查人物重影、头身和四肢对齐、遮挡边界及背景变形，并如实填写页面评分。页面默认 40 帧全部未评审。

启动**新的独立本地页面**（无需重启旧评审服务）：

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
/home/yy/conda_envs/camerahmr/bin/python scripts/fitting512/review_server.py --host 127.0.0.1 --port 18873
```

浏览器打开 `http://127.0.0.1:18873/`。左右分别是“原 MAMMA · 阈值 0.5”和“原 MAMMA＋人体 fitting · 阈值 0.5”；切帧先载入 1440×720 快速预览，放大后读取原分辨率。页面可导出／导入本次独立评审 JSON，不读取或覆盖旧 0.85 对照评分。此次完整清单在[comparison_gpu_trial2/index.json](../outputs/experiments/original_mamma_fitting_20261025/comparison_gpu_trial2/index.json)，其中逐人记录原共同模板点 ID、初始化相机、LM 状态、前后目标值、位移和网格准入结果。两套标定均已贯通。实际运行只验证了离散的 40 对图，不代表连续视频稳定性或最终合图算法表现。

## 用户人工观察（2026-10-25）

用户查看后反馈：**原 MAMMA 与原 MAMMA＋人体 fitting 的合图效果基本一致。** 这是本轮总体人工判断，不是已填写40帧逐项评分，也不代表有真实点位GT。当前40对离散图片与既定求解预算下，未观察到明显的额外合图视觉收益；fitting同时增加初始化和优化计算开销。44组均达到预设步数上限，不能将本轮结果外推为充分收敛后仍无收益，或所有场景中fitting都无效。

当前建议保留原MAMMA直接合图作为默认路径，fitting保留为可选实验分支；不依据本轮总体反馈自动追加迭代、训练或改部署。用户未逐帧评分的项目继续保持未评审。
