#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/paper.yaml}"
mkdir -p logs

python scripts/run_paper_suite.py \
  --config "$CONFIG" \
  --stages all \
  2>&1 | tee logs/object_context_paper_full.log

python scripts/check_paper_completion.py \
  --config "$CONFIG"

python scripts/analyze_paper_results.py \
  --config "$CONFIG" \
  2>&1 | tee logs/object_context_paper_analysis.log

echo "Complete paper outputs are under outputs/results/object_context_paper_v1/analysis/"
