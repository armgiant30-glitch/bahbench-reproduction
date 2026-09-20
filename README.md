# BAHBench 复现实验代码包

本代码包用于复现论文 **BAHBench: A Unified Benchmark for Evaluating Bio-Acoustic Health With Acoustic Foundation Models** 的可运行部分。

## 复现范围

当前包只包含论文版 Transformer 与 LoRA 秩扫描，不包含线性探针结果。

- 论文版 Transformer：冻结基础模型 → 768 维投影 → 单层标准 Transformer → 分类头
- LoRA 秩扫描：HuBERT-base 的 Q/K/V LoRA，r=2,4,8,16
- 任务：ICBHI、TORGO、SEP-28k

## 目录

```text
src/                             公共源码
experiments/01_paper_transformer/论文版 Transformer 实验
experiments/02_lora_rank_sweep/  LoRA 秩扫描实验
experiments/03_confusion_matrices/混淆矩阵实验
results_summary/                 已整理结果、CSV、图表和最终报告
DATA_AND_SPLIT.md                数据路径和论文 split 差异
environment.md                   运行环境和依赖
```

## 复现顺序

1. 按 `environment.md` 准备 Python、PyTorch、Transformers 和 GPU。
2. 按 `DATA_AND_SPLIT.md` 准备数据 manifest。
3. 运行 `experiments/01_paper_transformer/run.sh`。
4. 运行 `experiments/02_lora_rank_sweep/run.sh`。
5. 运行 `experiments/03_confusion_matrices/run.sh`。

## 代码约定

- 所有任务统一使用 16 kHz 音频。
- 论文版下游固定为 `d_model=768, nhead=8, num_layers=1`。
- Transformer 下游只允许一层，不是多层堆叠。
- seed 固定为 0。
- 数据、模型权重和输出路径通过脚本中的变量或命令行参数指定。
- 结果 JSON 会保存 `args`、`classes`、`split`、`metrics` 和 `predictions`。

## 结果说明

`results_summary/` 包含当前整理后的 Transformer 和 LoRA 结果。所有与论文的数值对比均标注为近似比较，因为当前数据版本、split 和音频截断尚未完全对齐论文。
