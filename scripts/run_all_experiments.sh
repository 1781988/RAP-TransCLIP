#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-standard}"
DATASETS=(AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19)

if [[ "$MODE" == "standard" ]]; then
  CONFIG="configs/standard.yaml"
  MODELS=(GeoRSCLIP)
  ARCHS=(ViT-L-14)
elif [[ "$MODE" == "full" ]]; then
  CONFIG="configs/full_matrix.yaml"
  MODELS=()
  ARCHS=()
else
  echo "Usage: bash scripts/run_all_experiments.sh [standard|full]"
  exit 2
fi

for dataset in "${DATASETS[@]}"; do
  python scripts/build_index.py \
    --dataset "$dataset" \
    --config "$CONFIG"
done

if [[ "$MODE" == "standard" ]]; then
  python scripts/run_all_standard.py \
    --config "$CONFIG" \
    --stage all \
    --datasets "${DATASETS[@]}" \
    --models "${MODELS[@]}" \
    --architectures "${ARCHS[@]}" \
    --methods zero_shot rs_transclip textgraph_transclip
  RESULT_ROOT="outputs/results/textgraph"
else
  python scripts/run_all_standard.py \
    --config "$CONFIG" \
    --stage all \
    --datasets "${DATASETS[@]}" \
    --methods zero_shot rs_transclip textgraph_transclip
  RESULT_ROOT="outputs/results/textgraph_full_matrix"
fi

python scripts/summarize_results.py \
  --input "$RESULT_ROOT/raw_results.csv" \
  --output "$RESULT_ROOT/summary.csv"

echo "Completed. See $RESULT_ROOT/raw_results.csv and summary.csv"
