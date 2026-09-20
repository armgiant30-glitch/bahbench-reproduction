# 运行环境

- Python: 3.8.10
- PyTorch: 2.0.0+cu118
- Transformers: 4.46.3
- NumPy: 1.24.2
- soundfile: 0.13.1
- GPU: NVIDIA GeForce RTX 3090 24 GB
- 音频读取采样率: 16 kHz
- 随机种子: 0

## 依赖

```bash
pip install torch torchaudio transformers soundfile numpy
```

## Hugging Face 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
```

## 当前论文版配置

```text
d_model = 768
nhead = 8
num_layers = 1
dim_feedforward = 3072
dropout = 0.5
batch_size = 32
epochs = 50
lr = 5e-4
weight_decay = 1e-2
LoRA targets = Q/K/V
LoRA ranks = 2,4,8,16
```
