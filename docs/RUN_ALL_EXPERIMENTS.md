# Complete Experiment Execution Guide

This document gives the exact order for generating every result used by the working RAP-TransCLIP paper draft. Dataset images are not included in the repository.

## 1. Install

```bash
git clone https://github.com/1781988/RAP-TransCLIP.git
cd RAP-TransCLIP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-compatible PyTorch build from pytorch.org first.
pip install -e .
pip install faiss-cpu
pytest -q
```

## 2. Copy datasets

Copy the ten RS-TransCLIP datasets to `datasets/<DATASET>/`. Every dataset must contain `classes.txt`, `class_changes.txt`, and `images/`.

## 3. Initial ten-dataset screening

```bash
python scripts/download_checkpoints.py --models GeoRSCLIP --architectures ViT-L-14
bash scripts/run_all_experiments.sh standard
```

This produces 30 main rows: ten datasets multiplied by three methods.

## 4. Main component and extended ablations

```bash
python scripts/run_ablation_suite.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --extended
```

The suite includes the RS-TransCLIP baseline, prompt-only variant, prompt-plus-prior variant, complete method, four prompt-score removals, and two graph variants.

## 5. Realistic target-distribution protocols

Run only after the full-dataset screening is stable:

```bash
python scripts/run_protocol_suite.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14
```

This evaluates 25%, 50%, and 75% active-class subsets and Dirichlet long-tail targets with alpha 0.1, 0.5, and 1.0.

## 6. Full VLM/backbone matrix

```bash
python scripts/download_checkpoints.py --models RemoteCLIP GeoRSCLIP SkyCLIP50
bash scripts/run_all_experiments.sh full
```

The full matrix covers CLIP RN50/B32/L14, RemoteCLIP RN50/B32/L14, GeoRSCLIP B32/L14/H14, and SkyCLIP50 B32/L14. It is expensive because every dataset/model/architecture combination requires image encoding.

## 7. Result files

- `outputs/results/raw_results.csv`: one row per run.
- `outputs/results/summary.csv`: grouped mean and standard deviation.
- `outputs/features/`: cached image and prompt embeddings.
- `outputs/indexes/`: validated dataset indexes.

Every result row includes `experiment_tag`, protocol metadata, solver diagnostics, and a configuration hash.

## 8. Clean rerun

The runner appends rows. For a completely fresh campaign:

```bash
mv outputs/results/raw_results.csv outputs/results/raw_results.backup.csv 2>/dev/null || true
rm -f outputs/results/summary.csv
```

Do not delete `outputs/features/` unless the dataset version, preprocessing, model checkpoint, or prompt bank changes.
