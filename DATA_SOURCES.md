# 数据来源与重建方式

本仓库只包含源码、配置、测试与文档。数据集与模型权重由 `data/` 下的脚本
按需下载，不纳入版本控制（`.gitignore` 中 `data/**`，仅保留目录结构与
`download_*.sh`）。

## 一键取得

所有下载脚本都要求先在对应网站注册账号，再在脚本内提供账号信息。
每个脚本都支持 `--help`。

```bash
# 演示序列（无需注册）
bash data/download_example.sh

# MAMMA 数据集（需注册 https://mamma.is.tue.mpg.de/）
bash data/download_mamma_dance.sh --meta --pred --videos
bash data/download_mamma_eval.sh  --gt --videos --masks
bash data/download_mamma_iphone.sh
bash data/download_mamma_multi_people.sh
bash data/download_mamma_syn_wd.sh

# 模型权重与身体模型
bash data/download_mamma_weights.sh --all   # MammaNet 关键点检测 checkpoint + 512 顶点矩阵
bash data/download_smplx_locked_head.sh     # SMPL-X locked-head（需 smpl-x.is.tue.mpg.de 账号）
```

## 不在仓库里的内容

| 路径 | 内容 | 取得方式 |
| --- | --- | --- |
| `data/` | MAMMA 数据集、视频、GT、mask、marker | `data/download_mamma_*.sh` |
| `data/body_models/` | SMPL-X 模型（含 `downsampled_verts/verts_512.pkl`） | `data/download_smplx_locked_head.sh` + `download_mamma_weights.sh --verts` |
| `data/body_models/downsampled_verts/smplx_512_body_parts.npz` | 约 6 KB 派生资产，由 `landmarks/lib/datasets/_smplx_512_body_parts.json` 生成 | 按该 JSON 重新生成后 `git add -f` |
| `output/`、`outputs/`、`landmark_outputs/`、`checkpoints/` | 训练与推理产物、checkpoint | 由本仓库训练产出，不可下载 |
| `configs/sam2.1/` | SAM 2 配置 YAML | 见 `docs/INSTALL.md` 的手动 wget 回退步骤 |
| `.env`、`.env.local` | 本机环境覆盖与密钥 | 从 `.env.example` 复制后自行填写 |

## 许可提醒

MAMMA 数据与 SMPL-X 模型各自独立授权，均需在官网注册并同意条款。
账号凭证不要写进仓库，也不要提交下载后的数据 —— 上游条款不允许再分发。

## 重建顺序

```bash
micromamba create -f requirements/mamma_conda.yaml -y
micromamba activate mamma
bash data/download_example.sh      # 最小验证
bash data/download_mamma_weights.sh --all
bash data/download_smplx_locked_head.sh
python -m inference --task configs/examples/presets/quick.yaml \
    --capture configs/examples/captures/mamma_example.json
```
