# ObjectContext-CLIP

**Training-Free Multi-Scale Object–Context Collaborative Inference for Zero-Shot Remote-Sensing Scene Classification**

This repository contains the active ObjectContext-CLIP implementation. The repository name is retained for continuity with earlier experiments; the current paper is independent of RS-TransCLIP transductive optimization.

ObjectContext-CLIP combines:

- one whole-image feature for scene layout and environmental context;
- deterministic multi-scale local crops for small objects and fine structures;
- class-specific context descriptions and local object/structure cues;
- evidence-adaptive fusion without target-data training.

The image and text encoders remain frozen and every image is classified independently.

## 1. Controlled methods

The framework provides six controlled methods:

1. `global_classname`: whole-image zero-shot classification with class-name prompts;
2. `multicrop_classname`: whole-image and local views matched to the same class-name prompts;
3. `global_context`: whole-image classification with class names and scene-context descriptions;
4. `object_only`: local crops matched to class-specific object/structure cues;
5. `fixed_object_context`: fixed context/object fusion;
6. `object_context`: adaptive ObjectContext-CLIP inference.

The critical comparison is not against RS-TransCLIP. It is ObjectContext-CLIP versus whole-image zero-shot, simple multi-crop inference, context-only inference, object-only inference, and fixed fusion.

## 2. Frozen v2 method

The paper configuration is fixed in:

```text
configs/paper.yaml
```

The v2 method uses:

```yaml
inference:
  object_topk: 2
  object_view_topk: 2
  class_consensus_view_topk: 2
  consensus_power: 1.0
  class_consensus_power: 1.0
  object_concept_mode: correct
```

For every local cue, evidence is averaged over the strongest two local views instead of using one hard maximum. Every candidate class also receives its own multi-view consensus score. The frozen configuration must not be tuned using the seven validation datasets.

## 3. Development and validation protocol

The three pilot datasets were used for method development and ablation:

```text
AID
PatternNet
RESISC45
```

The following seven datasets are held out for frozen-configuration validation:

```text
EuroSAT
MLRSNet
OPTIMAL31
RSC11
RSICB128
RSICB256
WHURS19
```

All ten datasets are reported in the final main table, but the paper explicitly distinguishes development results from independent validation results.

### Completed three-dataset pilot

| Method | AID | PatternNet | RESISC45 | Average |
|---|---:|---:|---:|---:|
| Global-ClassName | 72.6400 | 76.5592 | 73.4095 | 74.2029 |
| MultiCrop-ClassName | 73.4800 | 74.6579 | 73.8730 | 74.0036 |
| Global-Context | 71.6300 | 79.0757 | 77.0825 | 75.9294 |
| Object-Only | 65.1000 | 60.1579 | 65.9651 | 63.7410 |
| Fixed Object-Context | 72.6600 | 76.1776 | 78.0825 | 75.6400 |
| **ObjectContext-CLIP** | **73.2500** | **77.4046** | **78.5746** | **76.4097** |

Average gains are +2.2068 over Global-ClassName, +2.4061 over MultiCrop-ClassName, +0.4803 over Global-Context, and +0.7697 over fixed fusion. Correct local concepts outperform shuffled concepts by 3.5365 points. These results justify frozen validation, but they do not establish universal class-level improvement; several classes still degrade substantially.

## 4. Repository layout

```text
RAP-TransCLIP/
├── configs/
│   ├── paper.yaml
│   ├── standard.yaml
│   ├── full_matrix.yaml
│   └── concepts/
├── paper/
│   └── ObjectContext_CLIP_Chinese_Draft.md
├── rap_transclip/
│   ├── feature_extraction.py
│   ├── object_context.py
│   ├── runner.py
│   └── ...
├── scripts/
│   ├── run_paper_suite.py
│   ├── analyze_paper_results.py
│   ├── run_refinement_suite.py
│   ├── analyze_object_context.py
│   └── ...
└── tests/
```

## 5. Environment

Recommended environment:

- Linux;
- Python 3.10 or 3.11;
- a CUDA GPU;
- PyTorch matching the CUDA driver;
- `open-clip-torch`.

```bash
cd /home/user/GMY/RAP-TransCLIP
conda activate rap-transclip
python -m pip install --upgrade pip setuptools wheel
pip install -e .
pytest -q
```

## 6. Dataset and checkpoint preparation

Dataset structure:

```text
datasets/<DATASET>/
├── classes.txt
├── class_changes.txt
└── images/
```

Build all indexes:

```bash
for d in AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19; do
  python scripts/build_index.py --dataset "$d" --config configs/paper.yaml
done
```

Required primary checkpoint:

```text
checkpoints/GeoRSCLIP/RS5M_ViT-L-14.pt
```

Cross-backbone experiments additionally require:

```text
checkpoints/RemoteCLIP/RemoteCLIP-ViT-L-14.pt
checkpoints/SkyCLIP50/SkyCLIP_ViT_L14_top50pct_epoch_20.pt
```

CLIP ViT-L/14 is loaded through `open_clip` and may download its pretrained weights on first use.

## 7. Feature cache

Clean multi-scale features are stored under:

```text
outputs/features_object_context/<dataset>/<model>/<architecture>/clean/
```

Each image has one whole-image view and ten deterministic local crops:

- scales: 0.50 and 0.75;
- positions: center and four corners.

The paper suite reuses cached features. Image features are re-extracted only when a dataset/backbone/resolution variant is missing or `--overwrite-features` is supplied.

## 8. Complete paper experiment suite

The complete suite is resumable. Existing result rows are skipped unless `--force-evaluate` is supplied.

### Preflight only

```bash
python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages preflight
```

### One-command full run

```bash
mkdir -p logs

python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages all \
  2>&1 | tee logs/object_context_paper_full.log
```

This runs:

1. ten-dataset GeoRSCLIP ViT-L/14 main experiment with six methods;
2. development-set local-view, cue-count, consensus, scale, and crop-count ablations;
3. ten-dataset correct/shuffled/generic concept controls;
4. 1×/2×/4×/8× resolution experiments on the three development datasets;
5. CLIP, RemoteCLIP, GeoRSCLIP, and SkyCLIP50 ViT-L/14 validation on the development datasets.

### Run stages separately

```bash
python scripts/run_paper_suite.py --config configs/paper.yaml --stages main
python scripts/run_paper_suite.py --config configs/paper.yaml --stages ablation
python scripts/run_paper_suite.py --config configs/paper.yaml --stages concepts
python scripts/run_paper_suite.py --config configs/paper.yaml --stages resolution
python scripts/run_paper_suite.py --config configs/paper.yaml --stages cross_backbone
```

Use cached features only:

```bash
python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages main ablation concepts \
  --skip-feature-extraction
```

Force re-evaluation without re-encoding images:

```bash
python scripts/run_paper_suite.py \
  --config configs/paper.yaml \
  --stages main \
  --skip-feature-extraction \
  --force-evaluate
```

## 9. Experiments included in the paper suite

### 9.1 Main ten-dataset table

Methods:

```text
global_classname
multicrop_classname
global_context
object_only
fixed_object_context
object_context
```

Primary paper table uses Global-ClassName, MultiCrop-ClassName, Global-Context, and ObjectContext-CLIP. Object-Only and fixed fusion are placed in the ablation table.

### 9.2 Ablation on development datasets

The suite evaluates:

- one, two, and three local views per cue;
- one, two, and three local cues per class;
- with and without class-specific consensus;
- 0.50-scale crops only;
- 0.75-scale crops only;
- center crops only;
- the full two-scale ten-crop configuration.

Single-scale and center-crop ablations select subsets of the cached ten local views; they do not re-encode images.

### 9.3 Concept controls

```text
correct
shuffled
generic
```

Correct class-to-cue mapping must outperform shuffled and generic controls before the paper attributes gains to semantic factorization.

### 9.4 Resolution robustness

The three development datasets are evaluated at:

```text
clean
2× downsampling
4× downsampling
8× downsampling
```

### 9.5 Cross-backbone validation

The focused cross-backbone experiment uses ViT-L/14 only:

```text
CLIP
RemoteCLIP
GeoRSCLIP
SkyCLIP50
```

The complete architecture matrix is not required unless the focused experiment is stable.

## 10. Analyze all paper results

After the suite finishes:

```bash
python scripts/analyze_paper_results.py \
  --config configs/paper.yaml
```

Outputs are written to:

```text
outputs/results/object_context_paper_v1/analysis/
```

Generated artifacts include:

```text
table_main_top1.csv
table_main_macro_f1.csv
table_main_ece.csv
table_development_validation.csv
table_ablation.csv
table_concept_controls.csv
table_resolution.csv
table_cross_backbone.csv
table_efficiency.csv
table_significance_per_dataset.csv
table_significance_across_datasets.csv
paper_results_summary.md
```

The statistical report includes paired bootstrap confidence intervals, exact McNemar tests per dataset, and a Wilcoxon signed-rank test across datasets.

## 11. Frozen validation decision

The generated `paper_results_summary.md` reports `PASS` only when the seven held-out validation datasets satisfy all of the following:

1. average ObjectContext gain over Global-ClassName is at least +1.0 point;
2. average gain over MultiCrop-ClassName is at least +0.5 point;
3. average gain over Global-Context is positive;
4. at least five of seven datasets are non-negative versus Global-ClassName;
5. no validation dataset drops by more than 5.0 points versus Global-ClassName.

A `REVIEW` result does not mean the code failed. It means the paper must use a conditional claim, revise the method, or stop before submission.

## 12. Result interpretation

The strongest acceptable result pattern is:

- the seven validation datasets retain positive average gains;
- correct concept mapping is consistently better than shuffled concepts;
- adaptive fusion exceeds fixed fusion;
- two- or three-view cue pooling is more stable than one-view hard pooling;
- ObjectContext-CLIP remains competitive as resolution decreases;
- the same qualitative trend appears on at least three VLM backbones.

The paper must also report failure cases. Existing pilot failures include AID `viaduct`, PatternNet `christmas tree farm`, and RESISC45 `sparse residential`. These cases indicate semantic-description mismatch and must not be hidden by mean accuracy.

## 13. Paper

The active manuscript is:

```text
paper/ObjectContext_CLIP_Chinese_Draft.md
```

The manuscript includes completed pilot numbers and leaves full-suite values as placeholders until they are generated by `analyze_paper_results.py`.

## 14. License

Repository code is MIT licensed. Dataset and pretrained-model licenses remain with their respective owners.
