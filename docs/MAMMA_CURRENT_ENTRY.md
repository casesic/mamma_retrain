# 当前 MAMMA 训练入口

正常数据入口是 `mamma_ext.data.MammaSavedGTDataset`。它读取 `mamma_bundle_v1` 清单，以及包内相对路径所指向的 RGB、目标人物 mask、`mamma_512.npz` 和 `annotations.npz`。框只由 mask 按 `mask_bbox_inclusive_v1` 派生。训练端不读取原身体顶点、面片、动作时间文件或场景矩阵；生成端已完成资产几何检查。reader 仍核对模板点序、人物、帧、相机、整数时间、ROI、标签数值和文件存在性。

生成器位于相邻项目 `human_data_generator`。先完成一批当前格式数据，再将 [reader 配置](../configs/retrain/mamma_saved_gt_reader.json) 中的 `manifest` 改为该任务的 `observations.json` 路径；不支持历史清单的自动转换。当前默认路径只是未来任务 `key_batch_001` 的约定位置，在生成前不存在。

[训练配置](../configs/retrain/mamma_regular.json) 使用坐标与 uncertainty，并在标签合格后监督 visibility 的 0/1；-1 跳过 visibility loss，但不自动取消有效坐标监督。人与人及地面接触 loss 关闭；兼容原权重的头保留冻结，由训练适配处理。当前 `execution_authorized=false` 且 `label_qualification` 仍待可信负例，因此默认只能在有数据后做计划与数据检查，不能执行训练。

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
GEN_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
CUDA_VISIBLE_DEVICES='' PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 $GEN_PY scripts/mamma_retrain.py plan
CUDA_VISIBLE_DEVICES='' PYTHON_DOTENV_DISABLED=1 OPENBLAS_NUM_THREADS=1 $GEN_PY scripts/mamma_retrain.py check-data --limit 1
```

`--manifest` 可为某次计划指定另一个当前格式清单。拆分训练/验证时，episode、同一资产 ID 和同一资产 manifest ID 不得交叉。真实训练与评价需要以后单独授权的配置和显式 `--execute`；本轮没有构造模型、前向、更新或评价。
