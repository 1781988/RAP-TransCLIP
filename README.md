# RAP-TransCLIP

**Reliability-Aware Prompt and Active-Prior Transduction for Zero-Shot Remote Sensing Scene Classification**

RAP-TransCLIP is a training-free extension of RS-TransCLIP for remote-sensing vision-language models. It keeps the image and text encoders frozen and adds three test-time components:

1. **Category-wise prompt reliability estimation** instead of uniformly averaging all prompt templates.
2. **Active-class and non-uniform class-prior estimation** instead of assuming that every candidate class is present and balanced.
3. **Reliability-aware mutual-kNN propagation** to reduce error amplification on the transductive graph.

This repository is an experiment-ready research scaffold for the ten datasets used by RS-TransCLIP:

`AID`, `EuroSAT`, `MLRSNet`, `OPTIMAL31`, `PatternNet`, `RESISC45`, `RSC11`, `RSICB128`, `RSICB256`, and `WHURS19`.

> Current status: method implementation and experiment pipeline are provided; paper tables contain `TBD` placeholders until the experiments are run. No dataset images or pretrained checkpoints are included.

## Quick start

```bash
git clone https://github.com/1781988/RAP-TransCLIP.git
cd RAP-TransCLIP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -e .
pytest -q
```

Copy the ten datasets into `datasets/`, then run the standard experiment suite:

```bash
python scripts/download_checkpoints.py --models GeoRSCLIP --architectures ViT-L-14
bash scripts/run_all_experiments.sh standard
```

Detailed instructions are available in `docs/RUN_ALL_EXPERIMENTS.md`.
