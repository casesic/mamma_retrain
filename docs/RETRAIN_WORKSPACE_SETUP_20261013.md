# A1 接手与工作区记录（2026-10-13）

状态：独立本地 clone、五槽数据适配与 CPU 验收已完成。仅 `mamma_retrain` 可写；原 MAMMA、`densekp_retrain`、部署仓库只读。本批不训练、不提交。

- 原仓库 `/home/yy/Repos/mamma`：`main`，HEAD `588492f18876e2ed6888b2d26929047cb6b575e7`，remote `https://github.com/cuevhv/mamma.git`。
- 新目录从原本地仓库 `git clone --local --no-hardlinks --no-checkout` 建立，禁用 LFS smudge 后检出 HEAD；独立 `.git`，clone 的 origin 是本地原仓库。未拉网络、未复制 `.venv`、权重、身体、数据或输出。
- 原未提交项：`README.md` 尾部无关路径；`landmarks/lib/datasets/vitdet_dataset.py` 部署局部裁剪；三个 `scripts/{check_mammanet_local,demo_mammanet_local,test_densekp_delivery}.py` 是本地检查/演示。只迁入 `vitdet_dataset.py`，用于保持当前部署输入合同；原始 diff 见 `docs/source_vitdet_dataset_uncommitted.patch`。其余未迁入。
- 资源通过配置中的绝对只读路径引用；新输出限 `outputs/`。


## 环境与引用

新副本完整保留上游 `LICENSE`、`landmarks/train.py`、配置和原推理入口；新代码只增 ROI pilot Dataset/检查脚本及配置。原克隆无工作树软链接，`git rev-parse --git-dir` 为新目录自身 `.git`。未复制或链接整套 `data/`、`outputs/`。配置显式引用原 `verts_512.pkl`、原 checkpoint（仅元数据脚本）、densekp_retrain 审计/ROI/身体和本地 locked-head 属性；这些路径均只读。新结果写入 `mamma_retrain/outputs/acceptance/mamma_data_adapter_20261013/`。没有依赖当前 shell 中预存变量。

现有 Python 为 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python`（PyTorch `2.0.0+cu118`）。用 `sys.path.insert(0, '/mnt/data1/yy/home/yy/Projects/mamma_retrain/landmarks')` 并先导入 `lib.models` 后，实际 adapter import 为 `/mnt/data1/yy/home/yy/Projects/mamma_retrain/landmarks/lib/datasets/mamma_roi_pilot.py`，实际 `lib.datasets.vitdet_dataset` 为新副本同名文件。上游 `lib.datasets`/`lib.models` 初始导入存在循环；检查脚本采用上述顺序，不修原初始化。独立参考动态加载的是原仓库的 `vitdet_dataset.py`，不参与生产 Dataset。

可复现命令（cwd 均为 `/mnt/data1/yy/home/yy/Projects/mamma_retrain`）：

```bash
MPLCONFIGDIR=/tmp/mamma_mpl PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python -B scripts/check_mamma_roi_pilot.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python -B scripts/check_checkpoint_metadata_pilot.py
```

第一条输入 `configs/retrain/mamma_roi_pilot_20261013.json` 所列只读资源，输出 `outputs/acceptance/mamma_data_adapter_20261013/{checks,sample_manifest}.json`。第二条只从可信原 checkpoint 做 CPU 安全读取，输出同目录 `checkpoint_metadata.json`；当前安全反序列化遇 `omegaconf.listconfig.ListConfig` 因而未获得模型 key、shape、epoch/step。未实例化模型或改环境。

当前 `git status` 仅新副本内原部署裁剪文件修改、新配置、Dataset、检查脚本和文档。没有 commit 或 push；后续不得把本批输出当成正式训练资格。

## 里程碑

- 接手即记录原 HEAD、状态和迁入理由。
- 固定六槽、三份实际身体缓存及有序模板资格；五有框槽通过 ROI Dataset，物理域外槽保留无框原因。
- 活动部署 `fast_inference=True` 独立参考五例图像、mask、仿射精确一致；checkpoint 安全元数据因环境类型限制留待基线批。
