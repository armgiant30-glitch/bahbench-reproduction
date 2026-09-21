# BAHBench 复现结果摘要

本目录保存论文版 Transformer、LoRA 秩、池化方式和特征层选择的结构化结果。

## 文件

- `BAHBench_复现数据整理_最终版.docx`
- `BAHBench_复现数据整理_最终版.pdf`
- `data/transformer_results.csv`
- `data/lora_rank_results.csv`
- `BAHBench_复现论文_正式提交版_v0.3.2.docx/pdf`\n- `data/latest_pooling.csv`\n- `data/latest_data_size_rank.csv`\n- `data/latest_layer_sweep.csv`\n- `data/latest_layer_best.csv`\n- `figures/latest_pooling.png`\n- `figures/latest_data_size_rank.png`\n- `figures/latest_layer_sweep.png`

## 论文对照标记

- `接近论文`：WF1 与论文差异约在 ±2 个百分点内
- `低于论文`：当前 WF1 低于论文
- `高于论文但口径不同`：数值较高，但数据/split 不同
- 当前所有比较都受数据版本、官方 split 和音频截断影响
