#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
cd "$(dirname "$0")/../.."
for spec in \
  "ICBHI:/root/autodl-tmp/extra/manifest_icbhi.csv" \
  "TORGO:/root/autodl-tmp/extra/manifest_torgo.csv" \
  "SEP-28k:/root/autodl-tmp/bahbench-data/sep28k_manifest.csv"; do
  task=${spec%%:*}; manifest=${spec#*:}
  for r in 2 4 8 16; do
    python src/paper_transformer.py \
      --model facebook/hubert-base-ls960 --manifest "$manifest" \
      --mode lora --r "$r" --lora-targets qkv \
      --d-model 768 --nhead 8 --num-layers 1 --dim-feedforward 3072 \
      --dropout 0.5 --batch-size 32 --epochs 50 --lr 5e-4 --weight-decay 1e-2 \
      --max-seconds 6 --device cuda --out "results/paper_transformer_lora_${task}"
  done
done
