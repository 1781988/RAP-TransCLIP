#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/standard.yaml}"
MODEL="${MODEL:-GeoRSCLIP}"
ARCH="${ARCH:-ViT-L-14}"

mkdir -p logs

# 1. Focused pilot before spending resources on all datasets.
python scripts/run_all_standard.py \
  --config "$CONFIG" \
  --stage evaluate \
  --datasets AID RESISC45 RSICB128 \
  --models "$MODEL" \
  --architectures "$ARCH" \
  --methods zero_shot rs_transclip textgraph_transclip \
  2>&1 | tee logs/textgraph_pilot.log

python scripts/analyze_graph_edges.py \
  --config "$CONFIG" \
  --datasets AID RESISC45 RSICB128 \
  --model "$MODEL" \
  --architecture "$ARCH" \
  2>&1 | tee logs/textgraph_pilot_edges.log

# 2. The remaining stages should be launched only after checking the pilot.
echo "Pilot finished. Inspect outputs/results/textgraph/raw_results.csv"
echo "and outputs/results/textgraph/graph_edge_analysis.csv."
echo "Run the ten-dataset commands from README.md only if the pilot passes."
