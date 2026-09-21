#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
cd "$(dirname "$0")/../.."
for n in 50 100 200 500; do
  for r in 2 4 8 16; do
    out="results/data_size_rank_TORGO_n${n}"
    result="${out}/paper_transformer_lora_r${r}_facebook_hubert-base-ls960.json"
    if [[ -f "$result" ]]; then
      echo "[skip] ${result}"
      continue
    fi
    python src/paper_transformer.py \
      --model facebook/hubert-base-ls960 \
      --manifest /root/autodl-tmp/extra/manifest_torgo.csv \
      --mode lora --r "$r" --lora-targets qkv --max-per-class "$n" \
      --d-model 768 --nhead 8 --num-layers 1 --dim-feedforward 3072 \
      --dropout 0.5 --batch-size 32 --epochs 50 --lr 5e-4 --weight-decay 1e-2 \
      --max-seconds 6 --device cuda --out "$out"
  done
done
python src/plot_data_size_rank.py