# BAHBench 复现实验代码包

本代码包用于复现论文 **BAHBench: A Unified Benchmark for Evaluating Bio-Acoustic Health With Acoustic Foundation Models** 的可运行部分。

## 复现范围

当前主包只包含论文版 Transformer、LoRA 秩扫描、混淆矩阵和逐层特征实验，不包含线性探针结果。

- 论文版 Transformer：冻结基础模型 → 768 维投影 → 单层标准 Transformer → 分类头
- LoRA 秩扫描：HuBERT-base 的 Q/K/V LoRA，r=2,4,8,16
- 逐层特征选择：只改变基础模型 `--layer`，判断低中层/高层偏好
- 任务：ICBHI、TORGO、SEP-28k

## 目录

```text
src/                               公共源码
experiments/01_paper_transformer/  论文版 Transformer 实验
experiments/02_lora_rank_sweep/    LoRA 秩扫描实验
experiments/03_confusion_matrices/ 混淆矩阵实验
experiments/04_layer_sweep/        逐层特征选择实验
results_summary/                   已整理结果、CSV、图表和报告
DATA_AND_SPLIT.md                  数据路径和论文 split 差异
LAYER_SWEEP.md                     论文中的层选择判定规则
environment.md                     运行环境和依赖
```

## 复现顺序

1. 按 `environment.md` 准备 Python、PyTorch、Transformers 和 GPU。
2. 按 `DATA_AND_SPLIT.md` 准备数据 manifest。
3. 运行 `bash experiments/01_paper_transformer/run.sh`。
4. 运行 `bash experiments/02_lora_rank_sweep/run.sh`。
5. 运行 `python src/plot_rank_sweep.py`。
6. 运行 `bash experiments/03_confusion_matrices/run.sh`。
7. 如需复现层选择结论，运行 `bash experiments/04_layer_sweep/run.sh`。

## 代码约定

- 所有任务统一使用 16 kHz 音频。
- 论文版下游固定为 `d_model=768`、`nhead=8`、`num_layers=1`。
- Transformer 下游只使用一层，不是多层堆叠。
- 随机种子固定为 0。
- 结果 JSON 保存 `args`、`classes`、`split`、`metrics` 和 `predictions`。

## 结果说明

`results_summary/` 包含当前整理后的 Transformer、LoRA、混淆矩阵和论文对比结果。当前数据版本和 split 尚未完全对齐论文，因此所有论文数值比较都是近似比较。
