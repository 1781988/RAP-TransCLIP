# ObjectContext-CLIP

**Context-Anchored Local Object Evidence for Zero-Shot Remote-Sensing Scene Classification**

This repository implements a training-free remote-sensing scene classifier built around one central idea:

> Use the global context branch as the base classifier, and allow local object evidence to provide only selective residual corrections inside the context-supported candidate set.

The repository name is retained for continuity. The current work is independent of RS-TransCLIP and does not use transductive graph optimization.

## 1. Method

For each image, the framework computes:

- a whole-image context score from class names and scene descriptions;
- a local object score from deterministic crops and class-specific object/structure phrases;
- a class-specific multi-view consensus score.

The final method does not replace the context score. It applies:

```text
final score = context score + selective positive local residual
```

The local residual is restricted to the Top-M classes proposed by the context branch. This prevents an unrelated local texture from introducing an unsupported class and prevents local evidence from directly suppressing a context score.

Default configuration:

```yaml
inference:
  context_candidate_topk: 5
  positive_residual_only: true
  residual_weight: 0.50
  object_topk: 2
  object_view_topk: 2
  class_consensus_view_topk: 2
  fusion_temperature: 1.0
```

The paper treats this context-anchored residual rule as the single methodological contribution. Multi-view pooling and the concept bank are implementation components and are evaluated through controlled ablations.

## 2. Comparisons

The main table contains only external or controlled baselines:

1. `global_classname`;
2. `multicrop_classname`;
3. `global_context`;
4. `object_only`;
5. `fixed_object_context`;
6. `object_context` — the proposed context-anchored method.

Earlier in-house exploratory versions are not treated as competing methods and are not included in the paper main table.

## 3. Data and models

Ten supported datasets:

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

Existing image-feature caches remain compatible because the crop layout, text bank, and backbone interfaces have not changed.

## 4. Install

```bash
cd /home/user/GMY/RAP-TransCLIP
conda activate rap-transclip
pip install -e .
pytest -q
```

Expected package version:

```text
0.5.0
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

The suite is resumable and skips completed result keys. It runs:

- ten-dataset main comparison;
- five mechanism ablations on AID, PatternNet, and RESISC45;
- correct, shuffled, and non-class-specific object concept controls;
- 1×, 2×, 4×, and 8× resolution experiments;
- four ViT-L/14 backbones;
- classwise rescue/damage analysis;
- paired bootstrap confidence intervals, exact McNemar tests, and Wilcoxon tests.

The five ablations are:

```text
no context candidate restriction
context candidate Top-3
context candidate Top-10
signed residual instead of positive residual
no class-consensus weighting
```

They test the components of the final method. They are not comparisons with an earlier version.

## 7. Reuse existing feature caches

The new result directory is:

```text
outputs/results/object_context_anchor_v1/
```

The feature directory remains:

```text
outputs/features_object_context/
```

To run only inference using existing caches:

```bash
python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages main ablation concepts \
  --skip-feature-extraction
```

Resolution variants and missing backbones still require feature extraction.

## 8. Outputs

```text
outputs/results/object_context_anchor_v1/
├── raw_results.csv
├── predictions/
├── classwise_analysis_refined.csv
├── semantic_group_analysis_refined.csv
├── missing_paper_experiments.csv
└── analysis/
    ├── table_main_top1.csv
    ├── table_main_macro_f1.csv
    ├── table_main_ece.csv
    ├── table_ablation.csv
    ├── table_concept_controls.csv
    ├── table_resolution.csv
    ├── table_cross_backbone.csv
    ├── table_efficiency.csv
    ├── table_significance_per_dataset.csv
    ├── table_significance_across_datasets.csv
    └── paper_results_summary.md
```

Completion check:

```bash
python scripts/check_paper_completion.py --config configs/paper.yaml
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

The manuscript does not compare against an earlier internal ObjectContext version. Main claims are based on the six controlled methods, current ablations, cross-backbone results, and statistical analysis.

## 10. License

Repository code is MIT licensed. Dataset and pretrained-model licenses remain with their respective owners.
