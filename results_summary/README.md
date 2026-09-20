# BAHBench 复现结果摘要

本目录保存论文版 Transformer 与 LoRA 秩扫描的结构化结果。

## 文件

- `BAHBench_复现数据整理_最终版.docx`
- `BAHBench_复现数据整理_最终版.pdf`
- `data/transformer_results.csv`
- `data/lora_rank_results.csv`
- `figures/`：论文对照、LoRA 曲线、ΔWF1 热力图、数据就绪度图

## 论文对照标记

- `接近论文`：WF1 与论文差异约在 ±2 个百分点内
- `低于论文`：当前 WF1 低于论文
- `高于论文但口径不同`：数值较高，但数据/split 不同
- 当前所有比较都受数据版本、官方 split 和音频截断影响
