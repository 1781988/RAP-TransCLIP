#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/paper.yaml}"
mkdir -p logs

python scripts/run_paper_suite.py \
  --config "$CONFIG" \
  --stages all \
  2>&1 | tee logs/object_context_uncertainty_full.log

python scripts/check_paper_completion.py \
  --config "$CONFIG"

python scripts/analyze_object_context.py \
  --config "$CONFIG" \
  --datasets AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --experiment-tag uncertainty_main_georsclip \
  2>&1 | tee logs/object_context_uncertainty_classwise.log

python scripts/analyze_paper_results.py \
  --config "$CONFIG" \
  2>&1 | tee logs/object_context_uncertainty_analysis.log

echo "Complete outputs are under outputs/results/object_context_uncertainty_v1/"
