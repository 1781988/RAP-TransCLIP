# SA-RAP-TransCLIP

**Shift-Aware Reliability-Aware Prompt and Active-Prior Transduction for Zero-Shot Remote-Sensing Scene Classification**

This repository studies when transductive remote-sensing VLM adaptation should be enabled. The code contains four inference modes:

1. `zero_shot`: uniform-prompt zero-shot VLM;
2. `rs_transclip`: reproduced RS-TransCLIP-compatible solver;
3. `rap_transclip`: prompt-, prior-, and graph-adaptive solver;
4. `sa_rap_transclip`: shift-aware gated selection/fusion of RS-TransCLIP and RAP-TransCLIP.

The shift-aware method is motivated by the current empirical boundary: RAP-TransCLIP improves severe partial-class and long-tail protocols, but underperforms RS-TransCLIP on favorable full-class batches. SA-RAP estimates batch-level class-prior mismatch from unlabeled zero-shot predictions and conservatively selects or fuses the two experts.

No datasets or pretrained checkpoints are included.

## 1. Main changes in the shift-aware version

- Unsupervised batch shift estimator using effective-class compression, prior divergence, missing class evidence, and a confidence guard.
- `sa_rap_transclip` solver with RS-only, soft-fusion, and RAP-only execution branches.
- Prompt-reliability computation in chunks rather than materializing all `[prompt, image, class]` probabilities.
- Severe-shift component ablation script.
- Gate threshold/temperature sensitivity script.
- Diagnostic bundles containing gate statistics, priors, prompt weights, and assignments.
- Three-seed protocol runner.

## 2. Repository layout

```text
RAP-TransCLIP/
├── configs/
│   ├── standard.yaml
│   ├── full_matrix.yaml
│   └── prompts/rs106.txt
├── datasets/
├── checkpoints/
├── outputs/
├── paper/
│   └── SA_RAP_TransCLIP_Chinese_Draft.md
├── rap_transclip/
│   ├── reliability.py
│   ├── shift.py
│   ├── solver.py
│   └── runner.py
├── scripts/
│   ├── run_all_standard.py
│   ├── run_protocol_suite.py
│   ├── run_shift_component_ablation.py
│   ├── run_gate_sensitivity.py
│   └── run_paper_experiments.sh
└── tests/test_smoke.py
```

## 3. Environment

```bash
cd /home/user/GMY/RAP-TransCLIP

conda create -n rap-transclip python=3.10 -y
conda activate rap-transclip

python -m pip install --upgrade pip setuptools wheel

# Install the PyTorch build matching the server CUDA version first.
pip install torch torchvision

pip install -e .
pip install faiss-cpu

pytest -q
```

The current tests cover RS-TransCLIP, RAP-TransCLIP, SA-RAP-TransCLIP, probability normalization, prompt weights, and the expected increase of the shift score when classes are removed.

## 4. Dataset structure

```text
datasets/
├── AID/
│   ├── classes.txt
│   ├── class_changes.txt
│   └── images/
├── EuroSAT/
├── MLRSNet/
├── OPTIMAL31/
├── PatternNet/
├── RESISC45/
├── RSC11/
├── RSICB128/
├── RSICB256/
└── WHURS19/
```

Build indexes:

```bash
for d in AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19; do
  python scripts/build_index.py \
    --dataset "$d" \
    --config configs/standard.yaml
done
```

## 5. Checkpoints

Initial experiments use GeoRSCLIP ViT-L/14:

```bash
python scripts/download_checkpoints.py \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

The expected file is:

```text
checkpoints/GeoRSCLIP/RS5M_ViT-L-14.pt
```

## 6. Extract features

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage features \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

Features are cached under:

```text
outputs/features/<dataset>/<model>/<architecture>/
```

All methods use identical cached image and text embeddings.

## 7. Run one method

```bash
python scripts/run_experiment.py \
  --config configs/standard.yaml \
  --dataset EuroSAT \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --method sa_rap_transclip
```

The result CSV contains shift diagnostics in the `diagnostics` field, including:

- `score`;
- `gate`;
- `effective_class_ratio`;
- `prior_divergence`;
- `active_class_ratio`;
- `mean_confidence`;
- `solver_branch`.

## 8. Full-class comparison

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage evaluate \
  --models GeoRSCLIP \
  --architectures ViT-L-14 \
  --methods zero_shot rs_transclip rap_transclip sa_rap_transclip
```

This experiment tests whether SA-RAP restores the full-class performance lost by unconditional RAP adaptation.

## 9. Six shift protocols, three seeds

```bash
python scripts/run_protocol_suite.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --methods rs_transclip rap_transclip sa_rap_transclip \
  --seeds 1 2 3
```

Protocols:

- partial classes: 25%, 50%, 75%;
- Dirichlet long tail: alpha 0.1, 0.5, 1.0.

Run only the two severe protocols:

```bash
python scripts/run_protocol_suite.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --methods rs_transclip rap_transclip sa_rap_transclip \
  --seeds 1 2 3 \
  --severe-only
```

## 10. Severe-shift component ablation

```bash
python scripts/run_shift_component_ablation.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --seeds 1 2 3
```

The script compares:

- RS-TransCLIP;
- active prior only;
- class-wise prompts only;
- reliable graph only;
- prompt + prior;
- prior + graph;
- full RAP;
- shift-aware RAP.

It runs on partial-25 and long-tail alpha 0.1 because these are the regimes where the method is designed to help.

## 11. Gate sensitivity

Start with a reduced dataset subset:

```bash
python scripts/run_gate_sensitivity.py \
  --config configs/standard.yaml \
  --datasets EuroSAT AID RESISC45 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --thresholds 0.10 0.15 0.20 0.25 0.30 \
  --temperatures 0.02 0.04 0.08 \
  --seeds 1 2 3
```

Do not tune the threshold separately for each dataset. Select one global setting and report the complete sensitivity surface.

## 12. Paper experiment suite

After features have been extracted:

```bash
bash scripts/run_paper_experiments.sh configs/standard.yaml
```

This runs:

1. full-class four-method comparison;
2. all six shift protocols with three seeds;
3. severe-shift component ablation.

Gate sensitivity is intentionally separate because it is substantially larger.

## 13. Save diagnostic tensors

Set in YAML:

```yaml
runtime:
  save_diagnostics: true
```

or use:

```bash
--override runtime.save_diagnostics=true
```

Bundles are saved under:

```text
outputs/results/diagnostics/
```

They contain assignments, class priors, prompt weights, sample reliability, and gate statistics for later figures.

## 14. Memory controls

The main prompt-memory parameter is:

```yaml
prompt_reliability:
  prompt_chunk_size: 4
```

Test `16`, `8`, and `4` and report both peak memory and prediction agreement with the unchunked implementation. Smaller chunks reduce peak memory but may increase runtime.

Feature extraction memory is controlled separately:

```yaml
feature_extraction:
  batch_size: 64
```

## 15. Recommended experiment order

1. Run `pytest -q`.
2. Run EuroSAT smoke tests for all four methods.
3. Run full-class ten-dataset comparison.
4. Run the two severe protocols with three seeds.
5. Run all six protocols.
6. Run severe-shift component ablation.
7. Run gate sensitivity on three representative datasets.
8. Freeze the gate configuration.
9. Run four ViT-L/14 backbones.
10. Run the complete 11-backbone matrix only if the method remains stable.

## 16. Reproducibility constraints

- Ground-truth labels may construct controlled protocol subsets but must never enter the solver.
- Use the same cached features for every method.
- Do not select gate thresholds per dataset.
- Keep raw per-seed results.
- Report paired confidence intervals and corrected significance tests.
- Record the exact AID or Million-AID Level-2 version.
- Record the Git commit and YAML configuration for every result table.

## 17. Current scientific objective

The paper should not claim that RAP-TransCLIP universally improves RS-TransCLIP. The target claim is narrower:

> SA-RAP-TransCLIP detects severe class-prior mismatch from unlabeled target predictions, preserves RAP's gains when the shift is strong, and falls back to RS-TransCLIP when adaptation is unnecessary.

The working Chinese manuscript is:

```text
paper/SA_RAP_TransCLIP_Chinese_Draft.md
```

## License

Code in this repository is released under the MIT License. Dataset and pretrained-model licenses remain with their original providers.
