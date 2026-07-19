# ObjectContext-CLIP

**Context-Uncertainty-Gated Local Object Residuals for Zero-Shot Remote-Sensing Scene Classification**

This repository implements a training-free remote-sensing scene classifier built around one core rule:

> Keep the whole-image context classifier as the anchor, and increase local-object corrections only when the context prediction is uncertain.

The current work is independent of RS-TransCLIP and does not use transductive graph optimization.

## 1. Method

For every image, the framework computes:

- a whole-image context score from class names and scene descriptions;
- a local object score from deterministic crops and class-specific object/structure phrases;
- normalized entropy of the context prediction.

The final score is

```text
final score = context score + context uncertainty × positive local residual
```

The local branch cannot directly reduce the context score. A confident context prediction receives little correction; an ambiguous context prediction allows stronger local evidence.

Default configuration:

```yaml
inference:
  use_uncertainty_gate: true
  uncertainty_temperature: 1.0
  positive_residual_only: true
  residual_weight: 0.50
  object_topk: 2
  object_view_topk: 2
  fusion_temperature: 1.0
```

Multi-view cue pooling and the concept bank are supporting components rather than separate paper contributions.

## 2. Controlled comparisons

The main table contains:

1. `global_classname`;
2. `multicrop_classname`;
3. `global_context`;
4. `object_only`;
5. `fixed_object_context`;
6. `object_context` — the proposed uncertainty-gated residual method.

Earlier internal exploratory versions are not treated as baselines.

## 3. Data and backbones

Ten datasets:

```text
AID
EuroSAT
MLRSNet
OPTIMAL31
PatternNet
RESISC45
RSC11
RSICB128
RSICB256
WHURS19
```

Primary backbone:

```text
GeoRSCLIP ViT-L/14
```

Cross-backbone validation:

```text
CLIP ViT-L/14
RemoteCLIP ViT-L/14
GeoRSCLIP ViT-L/14
SkyCLIP50 ViT-L/14
```

Existing feature caches remain compatible because the image encoders, crop layout, prompt bank, and feature format are unchanged.

## 4. Install

```bash
cd /home/user/GMY/RAP-TransCLIP
conda activate rap-transclip
pip install -e .
pytest -q
```

Expected package version:

```text
0.6.0
```

## 5. Prepare indexes and checkpoints

```bash
for d in AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19; do
  python scripts/build_index.py --dataset "$d" --config configs/paper.yaml
done
```

```bash
python scripts/download_checkpoints.py \
  --models GeoRSCLIP RemoteCLIP SkyCLIP50 \
  --architectures ViT-L-14
```

Preflight:

```bash
python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages preflight
```

## 6. Complete experiment suite

One command:

```bash
bash scripts/run_paper_full.sh configs/paper.yaml
```

The suite is resumable and runs:

- ten-dataset main comparison;
- four focused ablations on AID, PatternNet, and RESISC45;
- correct, shuffled, and non-class-specific concept controls;
- 1×, 2×, 4×, and 8× resolution experiments;
- four ViT-L/14 backbones;
- classwise rescue/damage analysis;
- paired bootstrap confidence intervals, exact McNemar tests, and Wilcoxon tests.

The four ablations are:

```text
no uncertainty gate
signed residual instead of positive residual
single local view per cue
single local cue per class
```

They test the final method components and are not comparisons with an internal version.

## 7. Reuse existing feature caches

New result directory:

```text
outputs/results/object_context_uncertainty_v1/
```

Existing feature directory:

```text
outputs/features_object_context/
```

Run inference-only stages with cached features:

```bash
python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages main ablation concepts \
  --skip-feature-extraction
```

Resolution variants and missing backbones still require feature extraction.

## 8. Outputs

```text
outputs/results/object_context_uncertainty_v1/
├── raw_results.csv
├── predictions/
├── classwise_analysis_refined.csv
├── semantic_group_analysis_refined.csv
├── missing_paper_experiments.csv
└── analysis/
    ├── table_main_top1.csv
    ├── table_main_macro_f1.csv
    ├── table_ablation.csv
    ├── table_concept_controls.csv
    ├── table_resolution.csv
    ├── table_cross_backbone.csv
    ├── table_efficiency.csv
    ├── table_significance_per_dataset.csv
    ├── table_significance_across_datasets.csv
    └── paper_results_summary.md
```

ECE is not used as a paper decision metric because the controlled methods do not share a calibrated probability scale.

Completion check:

```bash
python scripts/check_paper_completion.py --config configs/paper.yaml
```

Expected unique rows:

```text
188
```

Analysis only:

```bash
python scripts/analyze_paper_results.py --config configs/paper.yaml
```

## 9. Paper

The active manuscript is:

```text
paper/ObjectContext_CLIP_Chinese_Draft.md
```

The manuscript reports only the final method, controlled baselines, focused ablations, and statistical analysis.

## 10. License

Repository code is MIT licensed. Dataset and pretrained-model licenses remain with their respective owners.
