# 实验二：LoRA 秩扫描

## 目的

在同一模型、同一数据和同一论文版 Transformer 下游下，只改变 LoRA 秩，观察性能和容量之间的关系。

## 对比点

- r=2、4、8、16
- WA、UA、WF1 随秩的变化
- ICBHI、TORGO、SEP-28k 的差距
- UA 是否随秩增大而退化

## 模型与代码

- 基础模型：`facebook/hubert-base-ls960`
- 代码：`src/paper_transformer.py`
- LoRA：`src/lora.py`
- 插入位置：Q/K/V

## 输出

每个任务、每个秩一个 JSON；文件名包含 `lora_r{r}`。
