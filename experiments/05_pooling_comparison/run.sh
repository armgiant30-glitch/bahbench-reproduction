#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
cd "$(dirname "$0")/../.."
for pool in cls mean max; do
  python src/paper_transformer.py \
    --model facebook/hubert-base-ls960 \
    --manifest /root/autodl-tmp/extra/manifest_torgo.csv \
    --mode frozen --pool "$pool" \
    --d-model 768 --nhead 8 --num-layers 1 --dim-feedforward 3072 \
    --dropout 0.5 --batch-size 32 --epochs 50 --lr 5e-4 --weight-decay 1e-2 \
    --max-seconds 6 --device cuda --out results/pooling_TORGO
done
python src/plot_pooling_comparison.py
