# BAHBench 复现检查清单

## 1. 环境

- Python 3.8+
- PyTorch 2.0+
- Transformers
- NumPy
- soundfile
- 可选 GPU：NVIDIA RTX 3090 或同级显卡

## 2. 数据

本代码包不包含音频数据集。需要先准备以下 manifest：

```text
path,label
```

当前脚本使用的云端 data manifest 在 `DATA_AND_SPLIT.md` 中列出。官方 split 版本需要额外提供：

```text
path,label,split
```

## 3. 运行

```bash
bash experiments/01_paper_transformer/run.sh
bash experiments/02_lora_rank_sweep/run.sh
python src/plot_rank_sweep.py
bash experiments/03_confusion_matrices/run.sh
bash experiments/04_layer_sweep/run_resume.sh
bash experiments/05_pooling_comparison/run.sh
bash experiments/06_data_size_rank_sweep/run.sh
```

## 4. 产物检查

- `results/` 下应出现 Transformer 和 LoRA JSON；
- JSON 应包含 `args`、`classes`、`split`、`metrics`、`predictions`；
- `confusion_matrices/` 下应出现 9 张分图和 `confusion_matrix_grid.png`；
- `plots/` 下应出现 LoRA 曲线图、池化对比图、样本量×秩热图和层选择图。

## 5. 目标对比检查

| 目标 | 判定方式 | 当前状态 |
|---|---|---|
| Transformer 主结构 | 使用 `paper_transformer.py` 而不是线性探针 | 已完成 |
| 模型规模 | base 与 large 同任务比较 | 部分完成 |
| LoRA 秩 | r=2,4,8,16 的 WF1 变化 | 已完成 |
| 池化方式 | cls/mean/max 同口径对比 | 已完成 TORGO 对照 |
| 样本量×秩 | 每类 50/100/200/500 × r=2/4/8/16 | 已完成 TORGO 扫描 |
| 层选择 | 1/4/7/10/12 五层 WF1/UA 对比 | 已完成三任务五层 |
| WA/UA 不平衡 | WA 与 UA 差距及少数类召回 | 已复现趋势 |
| 论文表 III 数值 | 官方数据 + 官方 split + 官方预处理后比较 | 未严格完成 |
| 完整模型矩阵 | 12 个基础模型配置 | 未完成 |
| LoRA 扩展秩 | r=32,64 | 未完成 |
| 手工特征 | eGeMAPS / ComParE | 未完成 |
| 六任务覆盖 | 6 个论文任务 | 未完成 |

## 6. 重要说明

当前复现使用固定 seed 0 的分层随机 split。官方 split 与完整数据版本尚未接入，因此现有结果是机制复现和近似数值复现，不应宣称为论文官方基准的严格复现。
