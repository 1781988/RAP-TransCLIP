# RAP-TransCLIP

**Reliability-Aware Prompt and Active-Prior Transduction for Zero-Shot Remote Sensing Scene Classification**

RAP-TransCLIP is a training-free extension of RS-TransCLIP for remote-sensing vision-language models. It keeps the image and text encoders frozen and adds three test-time components:

1. **Category-wise prompt reliability estimation** instead of uniformly averaging all prompt templates.
2. **Active-class and non-uniform class-prior estimation** instead of assuming that every candidate class is present and balanced.
3. **Reliability-aware mutual-kNN propagation** to reduce error amplification on the transductive graph.

This repository is an experiment-ready research scaffold for the ten datasets used by RS-TransCLIP:

`AID`, `EuroSAT`, `MLRSNet`, `OPTIMAL31`, `PatternNet`, `RESISC45`, `RSC11`, `RSICB128`, `RSICB256`, and `WHURS19`.

> Current status: method implementation and experiment pipeline are provided; paper tables contain `TBD` placeholders until the experiments are run. No dataset images or pretrained checkpoints are included.

## 1. Repository layout

```text
RAP-TransCLIP/
├── configs/
│   ├── standard.yaml
│   ├── full_matrix.yaml
│   └── prompts/rs106.txt
├── datasets/                     # copy datasets here; ignored by Git
│   └── <DATASET>/
│       ├── images/
│       ├── classes.txt
│       └── class_changes.txt
├── checkpoints/                  # copy/download VLM checkpoints here
├── outputs/                      # generated features and result CSV files
├── paper/
│   ├── ICASSP2027_RAP_TransCLIP_Draft.md
│   ├── RAP_TransCLIP_Chinese_Draft.tex
│   └── RAP_TransCLIP_Chinese_Draft.pdf
├── rap_transclip/
│   ├── config.py
│   ├── data.py
│   ├── feature_extraction.py
│   ├── graph.py
│   ├── metrics.py
│   ├── models.py
│   ├── prompts.py
│   ├── protocols.py
│   ├── reliability.py
│   ├── runner.py
│   └── solver.py
├── scripts/
│   ├── build_index.py
│   ├── download_checkpoints.py
│   ├── extract_features.py
│   ├── run_all_experiments.sh
│   ├── run_all_standard.py
│   ├── run_ablation_suite.py
│   ├── run_protocol_suite.py
│   ├── run_experiment.py
│   └── summarize_results.py
├── tests/test_smoke.py
├── requirements.txt
└── pyproject.toml
```

## 2. Environment setup

The recommended environment is Linux, Python 3.10 or 3.11, PyTorch 2.2 or newer, and a CUDA GPU.

```bash
git clone https://github.com/1781988/RAP-TransCLIP.git
cd RAP-TransCLIP

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install a PyTorch build matching your CUDA version first.
# Example only; use the command from pytorch.org for your machine.
pip install torch torchvision

pip install -e .
```

Optional but strongly recommended for large datasets:

```bash
pip install faiss-cpu
# For a compatible CUDA installation, faiss-gpu may be used instead.
```

Check the installation:

```bash
pytest -q
python -m rap_transclip.cli --help
```

## 2.1 Fast path: reproduce the complete ten-dataset screening

After the environment is installed, copy all ten datasets into `datasets/`, then run:

```bash
python scripts/download_checkpoints.py --models GeoRSCLIP --architectures ViT-L-14
bash scripts/run_all_experiments.sh standard
```

This executes the following fixed comparison on every dataset:

- inductive zero-shot VLM;
- the in-repository RS-TransCLIP-compatible baseline;
- the complete RAP-TransCLIP method.

The primary result files are `outputs/results/raw_results.csv` and `outputs/results/summary.csv`. Feature tensors are cached, so subsequent ablation runs do not re-encode images.

For the complete backbone/architecture matrix:

```bash
python scripts/download_checkpoints.py --models RemoteCLIP GeoRSCLIP SkyCLIP50
bash scripts/run_all_experiments.sh full
```

The full matrix is substantially more expensive. Complete the standard GeoRSCLIP ViT-L/14 screening first.

## 3. Dataset preparation

Do not commit datasets to Git. Copy each dataset into `datasets/<DATASET>/` using the same flat structure as RS-TransCLIP:

```text
datasets/EuroSAT/
├── classes.txt
├── class_changes.txt
└── images/
    ├── annualcropland_1.jpg
    ├── forest_1.jpg
    └── ...
```

`classes.txt` contains one machine-readable class token per line. `class_changes.txt` contains the corresponding display name used in text prompts. The two files must contain the same number of lines and preserve the same class order.

Example:

```text
# classes.txt
annualcropland
forest
highwayorroad

# class_changes.txt
annual crop land
forest
highway or road
```

The loader also accepts class-subdirectory layouts such as `images/forest/xxx.jpg`. For reproducibility, the flat RS-TransCLIP layout is preferred.

Build and validate an index for one dataset:

```bash
python scripts/build_index.py --dataset EuroSAT --config configs/standard.yaml
```

Build indexes for all ten datasets:

```bash
for d in AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19; do
  python scripts/build_index.py --dataset "$d" --config configs/standard.yaml
done
```

The command writes `outputs/indexes/<DATASET>.jsonl` and reports unmapped or unreadable files. Fix all mapping errors before feature extraction.

## 4. Pretrained model preparation

The default configuration contains four VLM families used by the original paper: CLIP, RemoteCLIP, GeoRSCLIP, and SkyCLIP50.

CLIP can be loaded directly through OpenCLIP. The remote-sensing checkpoints should be copied to the locations below or changed in `configs/standard.yaml`:

```text
checkpoints/
├── RemoteCLIP/
│   ├── RemoteCLIP-RN50.pt
│   ├── RemoteCLIP-ViT-B-32.pt
│   └── RemoteCLIP-ViT-L-14.pt
├── GeoRSCLIP/
│   ├── RS5M_ViT-B-32.pt
│   ├── RS5M_ViT-L-14.pt
│   └── RS5M_ViT-H-14.pt
└── SkyCLIP50/
    ├── SkyCLIP_ViT_B32_top50pct_epoch_20.pt
    └── SkyCLIP_ViT_L14_top50pct_epoch_20.pt
```

Checkpoint key prefixes are normalized automatically for common `module.` and `model.` wrappers.

### Recommended staged model plan

1. `GeoRSCLIP / ViT-L-14` on `EuroSAT`, `AID`, and `RESISC45`.
2. Add the other seven datasets after the first three reproduce sensible zero-shot accuracy.
3. Add `RemoteCLIP / ViT-L-14` and `SkyCLIP50 / ViT-L-14`.
4. Run the complete architecture matrix only after the method is stable.

## 5. Prompt bank

`configs/prompts/rs106.txt` contains the 106 remote-sensing templates used in the source framework. Each line is a template prefix; the class name is appended automatically.

Uniform RS-TransCLIP uses the normalized mean of all prompt embeddings. RAP-TransCLIP preserves the full tensor:

```text
texts_all.pt: [num_prompts, num_classes, embedding_dim]
```

and estimates a separate prompt distribution for every class.

## 6. Feature extraction

Extract one dataset/model combination:

```bash
python scripts/extract_features.py \
  --config configs/standard.yaml \
  --dataset EuroSAT \
  --model GeoRSCLIP \
  --architecture ViT-L-14
```

Expected outputs:

```text
outputs/features/EuroSAT/GeoRSCLIP/ViT-L-14/
├── images.pt
├── labels.pt
├── texts_all.pt
├── texts_uniform.pt
├── classes.json
├── image_paths.json
└── metadata.json
```

The labels are stored only for evaluation. Neither RS-TransCLIP nor RAP-TransCLIP uses ground-truth labels during inference. The number of candidate classes is obtained from the text prototypes, not from test labels.

Extract all enabled combinations:

```bash
python scripts/run_all_standard.py --config configs/standard.yaml --stage features
```

To limit a run:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage features \
  --datasets EuroSAT AID RESISC45 \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

## 7. Run the three primary methods

### 7.1 Zero-shot VLM

```bash
python scripts/run_experiment.py \
  --config configs/standard.yaml \
  --dataset EuroSAT \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --method zero_shot
```

### 7.2 RS-TransCLIP-compatible baseline

```bash
python scripts/run_experiment.py \
  --config configs/standard.yaml \
  --dataset EuroSAT \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --method rs_transclip
```

### 7.3 RAP-TransCLIP

```bash
python scripts/run_experiment.py \
  --config configs/standard.yaml \
  --dataset EuroSAT \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --method rap_transclip
```

RAP-TransCLIP performs category-wise prompt reliability scoring, weighted text-prototype construction, sample reliability estimation, mutual-kNN graph construction, and alternating updates of assignments, Gaussian prototypes, covariance, prompt weights, and class priors.

## 8. Run all ten standard-dataset experiments

```bash
python scripts/run_all_standard.py --config configs/standard.yaml --stage evaluate
```

Run only the initial ten-dataset screening with GeoRSCLIP ViT-L/14:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage evaluate \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

Results are appended to `outputs/results/raw_results.csv` and summarized with:

```bash
python scripts/summarize_results.py \
  --input outputs/results/raw_results.csv \
  --output outputs/results/summary.csv
```

## 9. Ablation experiments

```bash
python scripts/run_ablation_suite.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --extended
```

The suite covers uniform prompts, category-wise prompt reliability, active priors, complete RAP-TransCLIP, prompt-score terms, and graph variants.

## 10. Realistic protocols

```bash
python scripts/run_protocol_suite.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14
```

The protocol suite contains partial-class and Dirichlet long-tail evaluation.

## 11. Reproducibility checklist

1. Record the exact dataset source and version.
2. Record the class ordering and save `classes.json`.
3. Keep feature tensors fixed while comparing methods.
4. Run at least three seeds for graph/protocol randomness.
5. Confirm that labels are not passed to the solver.
6. Report both accuracy and macro-F1.
7. Report runtime and peak memory.
8. Save the full YAML and Git commit hash with every run.
9. Verify AID versus Million-AID naming.
10. Preserve raw per-run CSV files.

## 12. Paper workflow

The working paper is located at `paper/ICASSP2027_RAP_TransCLIP_Draft.md`. After experiments, replace every `TBD`, generate figures, select compact main results, and transfer the final text into the official ICASSP 2027 LaTeX template when released.

## 13. Attribution

This project is an independent research extension inspired by RS-TransCLIP and TransCLIP. The baseline implementation is rewritten for this repository. Cite the original papers and comply with their licenses when reusing checkpoints, prompt lists, or code.

## 14. License

Code in this repository is released under the MIT License. Dataset and pretrained-model licenses remain with their original providers.
