# ObjectContext-CLIP

**Training-Free Multi-Scale Object–Context Collaborative Inference for Zero-Shot Remote-Sensing Scene Classification**

This repository is the active implementation of **ObjectContext-CLIP**. The repository name is retained only for continuity with earlier experiments; the current method is independent of RS-TransCLIP transductive optimization.

ObjectContext-CLIP studies a remote-sensing classification problem that is not well represented by a single whole-image feature:

- global image features preserve scene layout and surrounding context;
- deterministic local crops preserve small objects and fine structures;
- class text is factorized into scene-context descriptions and local object/structure cues;
- an evidence-adaptive fusion rule combines the two branches without target training.

The image and text encoders remain frozen. Every image is classified independently.

## 1. Controlled inference methods

The evaluation framework provides six methods:

1. `global_classname`: whole-image zero-shot classification using class-name prompts;
2. `multicrop_classname`: whole-image and local views matched to the same class-name prompts;
3. `global_context`: whole-image classification using class names and scene-context descriptions;
4. `object_only`: local crops matched to class-specific object/structure cues;
5. `fixed_object_context`: fixed fusion of context and object evidence;
6. `object_context`: adaptive object–context collaboration.

The critical comparison is:

```text
ObjectContext-CLIP
vs. global zero-shot
vs. simple multi-crop zero-shot
vs. context-only and object-only controls
vs. fixed fusion
```

## 2. Refined v2 inference

The first pilot showed meaningful average gains but also exposed two weaknesses: a single local crop could dominate one cue, and the previous view-consensus statistic was shared by all classes. The refined implementation makes three targeted changes.

### 2.1 Multi-view cue pooling

For each class-specific cue, the score is averaged over the strongest `object_view_topk` local crops instead of taking one hard maximum. The default is:

```yaml
inference:
  object_view_topk: 2
```

This reduces accidental high responses from one crop.

### 2.2 Class-specific view consensus

Every candidate class receives its own multi-view support score. The adaptive route uses this score both for branch reliability and class-level gating:

```yaml
inference:
  class_consensus_view_topk: 2
  class_consensus_center: 0.0
  class_consensus_temperature: 0.50
  consensus_power: 1.0
  class_consensus_power: 1.0
```

### 2.3 Concept-bank negative controls

The runner supports three deterministic object-concept modes without re-encoding images:

```yaml
inference:
  object_concept_mode: correct   # correct | shuffled | generic
  concept_shuffle_seed: 17
```

- `correct`: use the released class-specific cue bank;
- `shuffled`: circularly permute cue banks across classes;
- `generic`: replace all class-specific cues with one global generic cue prototype.

Correct concepts must outperform these controls before the paper claims that local semantic factorization is effective.

## 3. Repository layout

```text
RAP-TransCLIP/
├── configs/
│   ├── standard.yaml
│   ├── full_matrix.yaml
│   └── concepts/
│       └── common_remote_sensing.yaml
├── datasets/
├── checkpoints/
├── outputs/
├── paper/
│   └── ObjectContext_CLIP_Chinese_Draft.md
├── rap_transclip/
│   ├── concepts.py
│   ├── data.py
│   ├── feature_extraction.py
│   ├── multiview.py
│   ├── object_context.py
│   ├── runner.py
│   └── ...
├── scripts/
│   ├── build_index.py
│   ├── build_concept_bank.py
│   ├── run_object_context_suite.py
│   ├── run_refinement_suite.py
│   ├── analyze_object_context.py
│   ├── summarize_refinement.py
│   ├── run_resolution_suite.py
│   └── run_all_standard.py
└── tests/test_smoke.py
```

The Python import package remains `rap_transclip` to avoid breaking the existing dataset and checkpoint utilities.

## 4. Environment

Recommended:

- Linux;
- Python 3.10 or 3.11;
- CUDA GPU;
- PyTorch matching the server CUDA driver;
- `open-clip-torch`.

```bash
cd /home/user/GMY/RAP-TransCLIP
conda activate rap-transclip
python -m pip install --upgrade pip setuptools wheel
pip install -e .
pytest -q
```

The current smoke suite checks deterministic views, multi-view cue pooling, class-specific consensus, bounded adaptive fusion, saved routing artifacts, and normalized output probabilities.

## 5. Dataset structure

Existing dataset folders can be reused:

```text
datasets/<DATASET>/
├── classes.txt
├── class_changes.txt
└── images/
```

Supported experiment names:

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

Build the pilot indexes when needed:

```bash
for d in AID PatternNet RESISC45; do
  python scripts/build_index.py \
    --dataset "$d" \
    --config configs/standard.yaml
done
```

## 6. Pretrained backbone

The focused experiment uses GeoRSCLIP ViT-L/14:

```bash
python scripts/download_checkpoints.py \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

Expected checkpoint:

```text
checkpoints/GeoRSCLIP/RS5M_ViT-L-14.pt
```

## 7. Concept bank

The common concept file is:

```text
configs/concepts/common_remote_sensing.yaml
```

Example:

```yaml
airport:
  group: object
  context:
    - an airport complex with long paved runways, taxiways, aprons, and terminal areas
  objects:
    - runway
    - airplane
    - taxiway
    - airport terminal
```

`group` is used only for offline analysis. It is not fed into the model.

Preview resolved concepts:

```bash
python scripts/build_concept_bank.py \
  --config configs/standard.yaml \
  --dataset AID
```

## 8. Multi-scale feature cache

Default views:

- crop scales: 0.50 and 0.75;
- positions: center and four corners;
- ten local crops plus one whole-image view.

Features are stored under:

```text
outputs/features_object_context/<dataset>/<model>/<architecture>/<variant>/
```

The refined v2 inference uses the same cached image and text features as the first ObjectContext pilot. **Do not re-extract features unless the crop layout, backbone, semantic prompts, or concept bank has changed.**

## 9. Re-run the refined three-dataset pilot

The new result root is separate from the previous pilot:

```text
outputs/results/object_context_refined_v2/
```

Run all six main methods and the five refinement controls using the cached features:

```bash
mkdir -p logs

python scripts/run_refinement_suite.py \
  --config configs/standard.yaml \
  --datasets AID PatternNet RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  2>&1 | tee logs/object_context_refinement_v2.log
```

The script runs:

```text
Main refined configuration:
  global_classname
  multicrop_classname
  global_context
  object_only
  fixed_object_context
  object_context

ObjectContext variants:
  view_topk1
  view_topk3
  no_class_consensus
  shuffled_object_concepts
  generic_object_concepts
```

No image encoding is performed by this script.

## 10. Analyze the refined pilot

Main comparison, semantic groups, class-level rescue/damage, and route diagnostics:

```bash
python scripts/analyze_object_context.py \
  --config configs/standard.yaml \
  --datasets AID PatternNet RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --experiment-tag object_context_refined_v2
```

Variant summary:

```bash
python scripts/summarize_refinement.py
```

Generated files:

```text
outputs/results/object_context_refined_v2/raw_results.csv
outputs/results/object_context_refined_v2/pilot_comparison_refined.csv
outputs/results/object_context_refined_v2/pilot_decision_refined.csv
outputs/results/object_context_refined_v2/refinement_comparison.csv
outputs/results/object_context_refined_v2/semantic_group_analysis_refined.csv
outputs/results/object_context_refined_v2/classwise_analysis_refined.csv
outputs/results/object_context_refined_v2/predictions/
```

Prediction bundles contain:

- probabilities and final scores;
- final class-specific object weights;
- standardized context and object scores;
- context reliability;
- object margin reliability;
- final object reliability;
- sample-level object branch weight;
- class-specific gate;
- class-specific view consensus.

## 11. Decision criteria before ten datasets

Continue to the ten-dataset experiment only when the refined pilot satisfies most of the following:

1. `object_context` exceeds `global_classname` by at least 2.0 points on average;
2. it exceeds `multicrop_classname` by at least 1.5 points;
3. it exceeds `global_context` by at least 0.5 point;
4. it exceeds `fixed_object_context` by at least 0.5 point;
5. correct object concepts outperform shuffled and generic concepts;
6. AID object-group degradation is substantially reduced;
7. local rescue exceeds local damage on at least two datasets;
8. `object_view_topk=2` or 3 is more stable than one-view hard pooling;
9. class-specific consensus provides a measurable benefit.

A method that only beats whole-image zero-shot but not multi-crop or context-only controls does not provide sufficient evidence for the proposed mechanism.

## 12. Ten-dataset experiment

After freezing the concept bank and all inference parameters:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage evaluate \
  --models GeoRSCLIP \
  --architectures ViT-L-14 \
  --methods \
    global_classname \
    multicrop_classname \
    global_context \
    object_only \
    fixed_object_context \
    object_context \
  2>&1 | tee logs/object_context_ten_datasets_v2.log
```

If features for the remaining datasets do not yet exist, run `--stage all` instead of `--stage evaluate`.

## 13. Resolution and cross-backbone experiments

Resolution suite:

```bash
python scripts/run_resolution_suite.py \
  --config configs/standard.yaml \
  --datasets AID PatternNet RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --factors 1 2 4 8
```

Cross-backbone validation should start with ViT-L/14 only:

```bash
python scripts/run_all_standard.py \
  --config configs/full_matrix.yaml \
  --stage all \
  --datasets AID PatternNet RESISC45 \
  --models CLIP RemoteCLIP GeoRSCLIP SkyCLIP50 \
  --architectures ViT-L-14 \
  --methods global_classname multicrop_classname global_context object_context
```

Do not run the complete architecture matrix until the refined pilot and focused cross-backbone experiment are stable.

## 14. Paper

The active Chinese manuscript is:

```text
paper/ObjectContext_CLIP_Chinese_Draft.md
```

Do not insert estimated results. Fill tables only from generated CSV files.

## 15. License

Repository code is MIT licensed. Dataset and pretrained-model licenses remain with their respective owners.
