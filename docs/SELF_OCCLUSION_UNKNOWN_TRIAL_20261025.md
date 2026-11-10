# 一次自遮挡 unknown 补负例试验

这次只试一条规则：对**原来为 -1**、且身体首射线与保存的渲染前景都指向“同一个人的身体挡住该点”的点，改为 visibility=0。原数据包、生成器默认标签策略和既有交付模型没有改动。这不是把所有 unknown 判成遮挡，也不是对所有新增负例逐点人工认证。

## 标签怎么改

[转换脚本](../scripts/self_occlusion_trial.py)要求该点几何、完整 ERP 坐标、物理域和渲染 coverage 有效；GT 射线及身体首命中距离均有限且为正，二者差大于 4 厘米；渲染前景实例是目标人物，GT 比该前景远大于 4 厘米；3×3 邻域至少 5 个相同前景实例，并且邻域证据完整。它不依据旧 reason_code 单独判定。旧 0/1 和不满足全部条件的 -1 原样保留。新标签 policy 为 `mamma512_self_occlusion_evidence_4cm_trial_v1`，被提升的点使用新的 reason_code 8；旧 reason_code 0–7 不改。

独立[试验清单](../outputs/experiments/self_occlusion_unknown_trial_20261025/index.json)列出原标签来源和每包转换数，[28/7 划分](../outputs/experiments/self_occlusion_unknown_trial_20261025/split.json)保持原 35 个 episode。RGB、逐人 mask 和 annotations 在同一磁盘上硬链接原文件；新 `mamma_512.npz` 独立写入，从未硬链接后改写原标签。逐项核查了 2,560 份新旧标签：除了 `policy`、符合规则的 visibility 和对应 reason，其他数组连字节内容都相同；11,040 个复用文件的 inode 与原件一致。

| 划分 | observed | 原始 unknown | 规则改成 0 | 实际新增 visibility 负例监督 |
| --- | ---: | ---: | ---: | ---: |
| 训练 28 场景 | 3,255 | 1,113,710 | 593,650 | 593,650 |
| 验证 7 场景 | 814 | 282,887 | 156,179 | 156,179 |

本次恰好所有候选点也满足原 reader 的坐标有效／裁剪条件，所以原始转换数与真正进入 visibility loss 的新增量相同；这不是该规则对其他数据的保证。训练侧原有有效负例 127,174 点变为 720,824 点，含负例观察从 953 增至 3,230；可见点和坐标有效点数不变。验证侧旧已知遮挡 21,756 点，加上独立的新增自遮挡子集后是 177,935 点；下面主表仍只用**旧已知标签的 21,756 点**计算遮挡召回，避免换分母造成假改善。训练 3,245 条可更新观察及 10 条无监督排除与旧计划一致，且 3,245 条 observation ID 顺序逐条相同。新[冻结顺序](../outputs/experiments/self_occlusion_unknown_trial_20261025/sampling_plan.json)有自己的 plan_id。

抽看了[单人近景](../outputs/experiments/self_occlusion_unknown_trial_20261025/inspection/batch_city_one_cycle_v0001_f8_rig_side_cam2_p1.png)、[双人后方目标](../outputs/experiments/self_occlusion_unknown_trial_20261025/inspection/batch_city_two_cycle_cross_v0001_f13_rig_side_cam2_p2.png)和[三人场景](../outputs/experiments/self_occlusion_unknown_trial_20261025/inspection/batch_city_three_cycle_straight_v0001_f7_rig_custom_1_cam2_p1.png)：红点为按 4 厘米双证据提升的候选，绿点为原可见点，灰点为仍保留的 unknown；图上每四个模板点抽画一个，避免遮住画面。这些例子的身体首命中及渲染前景差值最小分别为约 0.040–0.043 米，邻域计数至少为 5，未发现明显证据矛盾。二维画面本身无法逐点证明背面点位置，抽看不能代表全批认证。

## 一次训练与对照

[本轮配置](../configs/retrain/mamma_self_occlusion_trial_20261025.json)从原始 tensor_state 重新开始，沿用 batch 4、FP32、`class_mean`、AdamW 学习率 1e-5、原冻结策略和同一顺序。与旧全量配置逐字段对照，仅数据清单和新冻结计划路径不同。GPU0（`GPU-d481acc7-106f-29e5-1298-8c0d398261b2`）完成 812 次优化器更新：前 811 批各 4 条，最后一批 1 条；更新日志展开后与新旧计划的 3,245 个 observation ID **逐条一致**，没有重复、漏样或第二轮。训练循环 458.7 秒，loss 均有限，接触头保持冻结；最终[第 812 步 checkpoint](../outputs/experiments/self_occlusion_unknown_trial_20261025/train_full/checkpoint_update_000812.pt)仅供本实验。第 500 步文件是中间恢复点，没有按验证结果选权重。

主比较在**未修改的旧 7 场景、814 条验证标签**上进行，阈值统一 0.5。原模型和旧标签全量模型复用各自保存的预测，只对新模型前向。三列的观察顺序、模板点序、GT、坐标有效掩码与可见性原标签逐项一致。[完整统计](../outputs/experiments/self_occlusion_unknown_trial_20261025/comparison.json)保留七种场景分类。

| 模型 | 点均 ERP 误差 | 样本等权误差 | 可见召回 | 旧已知遮挡召回 | 旧遮挡误判可见 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 未微调原模型 | 4.3954 px | 5.3026 px | 82.64% | 95.73% | 929 / 21,756 |
| 原标签全量一遍 | 2.5242 px | 3.0956 px | 98.42% | 85.94% | 3,059 / 21,756 |
| 本次自遮挡试验全量一遍 | **2.4857 px** | **3.0641 px** | 95.53% | **91.18%** | **1,919 / 21,756** |

相对原标签全量模型，遮挡误判可见减少 1,140 点、遮挡召回提高 5.24 个百分点，代价是可见召回下降 2.89 个百分点；坐标误差略降。相对未微调原模型，点位和可见召回仍改善，但旧已知遮挡召回尚低 4.55 个百分点。这是一组固定设置的短试验，不是阈值搜索或独立真实 GT 测试。

另把**同一规则在验证集新增的 156,179 个点**单独作为子集，对三套已保存的预测按相同 0.5 阈值重算判遮挡比例：原模型 97.39%，原标签全量模型 85.39%，本次试验模型 92.80%。这里三列使用完全相同的新标签点集；它与上表旧已知遮挡分母分开，不能把改标签后总体召回的变化冒充模型改善。新增点的几何／渲染证据是自动规则，尚未逐点人工认证。

## 真实图片只供人工看图

新模型已在原先冻结的 40 对 ERP、319 个目标上完成推理。[预测数组](../outputs/experiments/self_occlusion_unknown_trial_20261025/predictions_real_self_occlusion_full_epoch.npz)和[页面数据](../outputs/experiments/self_occlusion_unknown_trial_20261025/review_predictions_self_occlusion_full_epoch.json)与原模型、旧全量模型的目标顺序、512 点模板、裁剪仿射一致；新模型输入指纹逐目标与已存旧输入一致，输出坐标和可见概率均有限。没有重做检测或修改 mask。按同一 0.5 阈值，旧全量模型预测可见 105,560 点，新模型预测可见 76,636 点；28,936 点从可见转为不可见，12 点反向变化。这只是**预测差异，不是真实准确率**。

在项目目录运行以下命令，并打开显示的网址。在“人工比较”中选择“原标签全量一遍 0.5 vs 自遮挡试验全量一遍 0.5”；新组与已有评分使用不同身份，初始 319 人全部未评审。当前已运行的旧服务不会自动加载新代码；下面的命令使用另一个端口，也不重启它。

```bash
cd /mnt/data1/yy/home/yy/Projects/mamma_retrain
/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python scripts/real_review_server.py --host 127.0.0.1 --port 18874
# 浏览器打开 http://127.0.0.1:18874/
```

隔离端口的页面与新模型资源已实际加载，首帧两路原图及 mask 均可读取；评审数据保持空白，未代用户打分。本轮不自动做双机合图，也不替换已交付的 0.85 微调版本。生成试验包、训练、旧标签评价、真实推理的入口分别为 `python -m scripts.self_occlusion_trial`、当前 `scripts/mamma_retrain.py train`、`python -m scripts.self_occlusion_eval_old_gt` 和 `python -m scripts.real_riding_self_occlusion_infer`；各脚本的具体输入和独立输出路径写在文件顶部，已有结果不可覆盖。
