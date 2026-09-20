#!/usr/bin/env bash
set -euo pipefail
bash experiments/01_paper_transformer/run.sh
bash experiments/02_lora_rank_sweep/run.sh
python src/plot_rank_sweep.py
bash experiments/03_confusion_matrices/run.sh
