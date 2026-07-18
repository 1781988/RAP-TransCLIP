# ObjectContext-CLIP

**Training-Free Multi-Scale Object–Context Collaborative Inference for Zero-Shot Remote-Sensing Scene Classification**

This repository is the active codebase for a new direction that is independent of RS-TransCLIP transductive optimization. The repository name is retained for continuity, but the current scientific framework is **ObjectContext-CLIP**.

The method studies a concrete remote-sensing problem:

- global image features preserve scene layout and surrounding context;
- deterministic local crops preserve small objects and fine structures;
- class text is factorized into context descriptions and local object/structure cues;
- an evidence-adaptive fusion rule balances the two branches without target training.

The image and text encoders remain frozen. Every image is classified independently; no transductive test collection is required.

## 1. Main methods

The evaluation framework provides six controlled methods:

1. `global_classname`: whole-image zero-shot classification with class-name prompts;
2. `multicrop_classname`: global and local views matched to the same class-name prompts;
3. `global_context`: whole-image classification with class names plus scene-context descriptions;
4. `object_only`: local crops matched to class-specific object/structure cues;
5. `fixed_object_context`: fixed fusion of context and object evidence;
6. `object_context`: evidence-adaptive object–context collaboration.

The critical comparison is not against RS-TransCLIP. It is:

```text
ObjectContext-CLIP
vs.
global zero-shot
vs.
simple multi-crop zero-shot
vs.
context-only and object-only controls
```

## 2. Repository layout

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
│   ├── download_checkpoints.py
│   ├── run_all_standard.py
│   ├── run_object_context_suite.py
│   ├── run_resolution_suite.py
│   └── summarize_results.py
└── tests/
```

The Python package name `rap_transclip` is temporarily retained to avoid breaking existing dataset/checkpoint utilities. It no longer denotes the active paper method.

## 3. Environment

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

# Install the correct CUDA build of torch/torchvision first.
pip install torch torchvision

pip install -e .
pip install faiss-cpu   # optional; not required by ObjectContext-CLIP

pytest -q
```

CLI:

```bash
object-context-clip --help
```

## 4. Dataset structure

Existing ten-dataset folders can be reused:

```text
datasets/<DATASET>/
├── classes.txt
├── class_changes.txt
└── images/
```

Supported first-party experiment names:

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

Build indexes for the pilot datasets:

```bash
for d in AID PatternNet RESISC45; do
  python scripts/build_index.py \
    --dataset "$d" \
    --config configs/standard.yaml
done
```

Build all indexes:

```bash
for d in AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19; do
  python scripts/build_index.py \
    --dataset "$d" \
    --config configs/standard.yaml
done
```

## 5. Pretrained backbone

The first experiment uses GeoRSCLIP ViT-L/14:

```bash
python scripts/download_checkpoints.py \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

Expected checkpoint:

```text
checkpoints/GeoRSCLIP/RS5M_ViT-L-14.pt
```

## 6. Concept bank

The released common knowledge file is:

```text
configs/concepts/common_remote_sensing.yaml
```

Each class entry may contain:

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

`group` is only used for analysis; it is not fed into the adaptive fusion rule.

Dataset-specific overrides are optional:

```text
configs/concepts/AID.yaml
configs/concepts/PatternNet.yaml
configs/concepts/RESISC45.yaml
```

Preview the resolved concept bank:

```bash
python scripts/build_concept_bank.py \
  --config configs/standard.yaml \
  --dataset AID
```

Output:

```text
outputs/results/object_context/concept_banks/AID.json
```

Classes not covered by the common file receive deterministic fallback descriptions. Before paper experiments, inspect all fallback entries and add reproducible dataset-specific corrections when needed.

## 7. Multi-scale features

Default local views:

- crop scales: 0.50 and 0.75;
- positions: center and four corners;
- ten local crops per image;
- one whole-image global view.

Features are stored under a separate root and do not overwrite previous RS/TextGraph features:

```text
outputs/features_object_context/
└── <dataset>/<model>/<architecture>/<variant>/
    ├── global_images.pt
    ├── local_images.pt
    ├── labels.pt
    ├── class_texts.pt
    ├── context_texts.pt
    ├── object_texts.pt
    ├── object_mask.pt
    ├── concept_bank.json
    └── metadata.json
```

Feature extraction is substantially more expensive than global CLIP inference because every image is encoded in multiple deterministic crops.

Extract one dataset:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage features \
  --datasets AID \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

For memory pressure, reduce:

```bash
--override feature_extraction.batch_size=8
```

## 8. Pilot experiment

Run AID, PatternNet, and RESISC45:

```bash
python scripts/run_object_context_suite.py \
  --config configs/standard.yaml \
  --datasets AID PatternNet RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --stage all \
  2>&1 | tee logs/object_context_pilot.log
```

If the features have already been extracted:

```bash
python scripts/run_object_context_suite.py \
  --config configs/standard.yaml \
  --datasets AID PatternNet RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --stage evaluate \
  2>&1 | tee logs/object_context_pilot_eval.log
```

Results:

```text
outputs/results/object_context/raw_results.csv
outputs/results/object_context/predictions/
```

Summarize:

```bash
python scripts/summarize_results.py \
  --input outputs/results/object_context/raw_results.csv \
  --output outputs/results/object_context/summary.csv
```

## 9. Pilot decision criteria

Do not immediately run all ten datasets. Continue only when the pilot shows:

1. `object_context` improves over `global_classname` by at least 1.0 point on average;
2. `object_context` improves over `multicrop_classname` by at least 0.5 point on average;
3. at least two of the three datasets improve;
4. object-driven classes benefit more than context-driven classes;
5. adaptive fusion differs meaningfully from fixed fusion.

If the method beats global zero-shot but not simple multi-crop, the current semantic factorization is not yet sufficient.

## 10. Full ten-dataset experiment

After freezing the concept bank and all inference parameters:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage all \
  --models GeoRSCLIP \
  --architectures ViT-L-14 \
  --methods \
    global_classname \
    multicrop_classname \
    global_context \
    object_only \
    fixed_object_context \
    object_context \
  2>&1 | tee logs/object_context_ten_datasets.log
```

## 11. Resolution robustness

The script creates separate feature variants for clean, 2x, 4x, and 8x downsampling:

```bash
python scripts/run_resolution_suite.py \
  --config configs/standard.yaml \
  --datasets AID PatternNet RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --factors 1 2 4 8 \
  2>&1 | tee logs/object_context_resolution.log
```

Output rows are distinguished by `feature_variant`.

## 12. Cross-backbone validation

Enable all ViT-L/14 backbones in `configs/full_matrix.yaml`, download their checkpoints, then run:

```bash
python scripts/download_checkpoints.py \
  --models RemoteCLIP GeoRSCLIP SkyCLIP50

python scripts/run_all_standard.py \
  --config configs/full_matrix.yaml \
  --stage all \
  --datasets AID PatternNet RESISC45 \
  --models CLIP RemoteCLIP GeoRSCLIP SkyCLIP50 \
  --architectures ViT-L-14 \
  --methods global_classname multicrop_classname object_context
```

Do not run the full matrix until the pilot succeeds.

## 13. Important experimental controls

The paper must include:

- global class-name zero-shot;
- simple multi-crop class-name baseline;
- context-only branch;
- object-only branch;
- fixed fusion;
- adaptive fusion;
- shuffled object-concept negative control;
- scale and crop-count ablation;
- object/context/mixed class-group analysis;
- runtime, memory, and feature-cache size.

All methods must use the same frozen backbone and deterministic views.

## 14. Paper

The active Chinese manuscript is:

```text
paper/ObjectContext_CLIP_Chinese_Draft.md
```

The manuscript contains no invented ObjectContext-CLIP numbers. Fill tables only from the generated CSV files.

## 15. License

Repository code is MIT licensed. Dataset and pretrained-model licenses remain with their respective owners.
