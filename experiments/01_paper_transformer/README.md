# 实验一：论文版 Transformer

## 目的

复现论文的统一下游结构：

```text
冻结基础模型 → 投影到 768 维 → 单层标准 Transformer → 分类头
```

## 对比点

- HuBERT-base vs wav2vec2-base vs wav2vec2-large
- 当前结果 vs 论文表 III 的同模型结果
- 声学任务 ICBHI/TORGO vs 语义任务 SEP-28k

## 代码

- `src/paper_transformer.py`
- `src/lora.py`
- `src/common.py`
- `src/metrics.py`

## 输出

- 每个模型/任务一个 JSON
- JSON 包含 `args`、`classes`、`split`、`metrics`、`predictions`
