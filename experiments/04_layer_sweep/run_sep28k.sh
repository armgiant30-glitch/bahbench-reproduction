#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
cd "$(dirname "$0")/../.."
model=facebook/hubert-base-ls960
manifest=/root/autodl-tmp/bahbench-data/sep28k_manifest.csv
out=results/layer_sweep_SEP-28k
LAYERS=${LAYERS:-"1 4 7 10 12"}
if [[ ! -f "$manifest" ]]; then
  echo "[error] missing manifest: $manifest" >&2
  exit 2
fi
for layer in $LAYERS; do
  result="${out}/paper_transformer_frozen_layer${layer}_facebook_hubert-base-ls960.json"
  if [[ -f "$result" ]]; then
    echo "[skip] ${result}"
    continue
  fi
  python src/paper_transformer.py \
    --model "$model" --manifest "$manifest" --mode frozen --layer "$layer" \
    --d-model 768 --nhead 8 --num-layers 1 --dim-feedforward 3072 \
    --dropout 0.5 --batch-size 32 --epochs 50 --lr 5e-4 --weight-decay 1e-2 \
    --max-seconds 6 --device cuda --out "$out"
done
python src/plot_layer_sweep.py