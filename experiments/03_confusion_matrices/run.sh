#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python src/make_confusion_matrices.py
