#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/standard.yaml}"
MODEL="${MODEL:-GeoRSCLIP}"
ARCH="${ARCH:-ViT-L-14}"

mkdir -p logs

python scripts/run_all_standard.py \
  --config "$CONFIG" \
  --stage evaluate \
  --models "$MODEL" \
  --architectures "$ARCH" \
  --methods zero_shot rs_transclip rap_transclip sa_rap_transclip \
  2>&1 | tee logs/paper_full_class.log

python scripts/run_protocol_suite.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --architecture "$ARCH" \
  --methods rs_transclip rap_transclip sa_rap_transclip \
  --seeds 1 2 3 \
  2>&1 | tee logs/paper_shift_protocols.log

python scripts/run_shift_component_ablation.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --architecture "$ARCH" \
  --seeds 1 2 3 \
  2>&1 | tee logs/paper_shift_ablation.log

echo "Paper experiment suite finished."
echo "Raw results: outputs/results/raw_results.csv"
