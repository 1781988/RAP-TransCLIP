# TextGraph-TransCLIP

**Text-Guided Boundary-Preserving Graph Transduction for Zero-Shot Remote-Sensing Scene Classification**

TextGraph-TransCLIP is a focused, training-free extension of RS-TransCLIP. It keeps the frozen VLM, uniform 106-prompt text prototype, Gaussian mixture, text anchor, and alternating inference unchanged. The only proposed modification is the graph:

- RS-TransCLIP propagates assignments on a visual cosine kNN graph.
- TextGraph-TransCLIP uses zero-shot text posteriors to reduce the conductance of visually similar but semantically inconsistent edges.
- When text predictions are uncertain, the edge gate approaches one and falls back to the original visual graph.

No dataset images or pretrained checkpoints are included.

## 1. Scientific question

Remote-sensing classes often share local texture, scale, and spatial patterns. A visual kNN graph can therefore connect different semantic classes and propagate incorrect pseudo-labels. This repository tests one narrow hypothesis:

> Can frozen VLM text posteriors act as boundary evidence for the visual graph and reduce cross-class propagation without changing the remaining RS-TransCLIP solver?

The repository intentionally does not combine prompt learning, class-prior adaptation, solver selection, or online memory with the proposed method. This keeps the contribution and ablations attributable to graph construction.

## 2. Methods

The supported inference methods are:

- `zero_shot`: uniform-prompt zero-shot VLM;
- `rs_transclip`: reproduced RS-TransCLIP-compatible solver with a visual graph;
- `textgraph_transclip`: RS-TransCLIP with text-guided edge conductance.

For a visual neighbor edge `(i,j)`, TextGraph uses the Hellinger affinity of the two zero-shot posterior distributions and entropy-derived node confidence. High-confidence semantic conflicts are suppressed; uncertain nodes retain the visual edge.

## 3. Repository layout

```text
RAP-TransCLIP/
├── configs/
│   ├── standard.yaml
│   ├── full_matrix.yaml
│   └── prompts/rs106.txt
├── datasets/                     # ignored by Git
├── checkpoints/                  # ignored by Git
├── outputs/                      # ignored by Git
├── paper/
│   └── TextGraph_TransCLIP_Chinese_Draft.md
├── rap_transclip/
│   ├── graph.py
│   ├── solver.py
│   ├── runner.py
│   └── ...
├── scripts/
│   ├── run_all_standard.py
│   ├── run_graph_ablation.py
│   ├── run_graph_sweep.py
│   ├── analyze_graph_edges.py
│   └── run_textgraph_experiments.sh
└── tests/test_smoke.py
```

## 4. Environment

The recommended environment is Linux, Python 3.10 or 3.11, PyTorch 2.2 or newer, and a CUDA GPU.

```bash
cd /home/user/GMY/RAP-TransCLIP

conda create -n rap-transclip python=3.10 -y
conda activate rap-transclip

python -m pip install --upgrade pip setuptools wheel

# Install the PyTorch wheel matching the server CUDA version first.
pip install torch torchvision

pip install -e .
pip install faiss-cpu

pytest -q
```

The smoke tests verify:

1. RS-TransCLIP and TextGraph output valid probability distributions;
2. `text_graph.semantic_strength=0` reproduces the RS solver;
3. confident semantic disagreement reduces edge conductance;
4. uncertain text predictions fall back to the visual graph.

## 5. Dataset structure

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

Build or refresh all indexes:

```bash
for d in AID EuroSAT MLRSNet OPTIMAL31 PatternNet RESISC45 RSC11 RSICB128 RSICB256 WHURS19; do
  python scripts/build_index.py \
    --dataset "$d" \
    --config configs/standard.yaml
done
```

## 6. GeoRSCLIP checkpoint and features

Download the initial backbone:

```bash
python scripts/download_checkpoints.py \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

Expected checkpoint:

```text
checkpoints/GeoRSCLIP/RS5M_ViT-L-14.pt
```

Extract all ten-dataset features:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage features \
  --models GeoRSCLIP \
  --architectures ViT-L-14
```

Existing feature files from the previous repository version remain compatible. TextGraph inference only requires `images.pt`, `labels.pt`, and `texts_uniform.pt`.

## 7. Required experiment order

### 7.1 Three-dataset pilot

Do not begin with the full model matrix. First run:

```bash
bash scripts/run_textgraph_experiments.sh configs/standard.yaml
```

Equivalent commands:

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage evaluate \
  --datasets AID RESISC45 RSICB128 \
  --models GeoRSCLIP \
  --architectures ViT-L-14 \
  --methods zero_shot rs_transclip textgraph_transclip

python scripts/analyze_graph_edges.py \
  --config configs/standard.yaml \
  --datasets AID RESISC45 RSICB128 \
  --model GeoRSCLIP \
  --architecture ViT-L-14
```

Inspect:

```text
outputs/results/textgraph/raw_results.csv
outputs/results/textgraph/graph_edge_analysis.csv
```

Recommended decision rule before continuing:

- the three-dataset mean Top-1 should exceed RS-TransCLIP by at least 0.5 percentage points;
- no pilot dataset should degrade by more than 1.0 point;
- weighted graph purity should increase or cross-class edge weight should decrease on the datasets that improve.

If these conditions fail, inspect the graph diagnostics before running more backbones.

### 7.2 Ten-dataset main experiment

```bash
python scripts/run_all_standard.py \
  --config configs/standard.yaml \
  --stage evaluate \
  --models GeoRSCLIP \
  --architectures ViT-L-14 \
  --methods zero_shot rs_transclip textgraph_transclip

python scripts/analyze_graph_edges.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14
```

### 7.3 Component ablation

```bash
python scripts/run_graph_ablation.py \
  --config configs/standard.yaml \
  --model GeoRSCLIP \
  --architecture ViT-L-14
```

Variants:

- original RS visual graph;
- visual × semantic graph without confidence fallback;
- full TextGraph;
- mutual-kNN TextGraph.

### 7.4 Neighborhood and conductance sweep

Run the pilot sweep first:

```bash
python scripts/run_graph_sweep.py \
  --config configs/standard.yaml \
  --datasets AID RESISC45 RSICB128 \
  --model GeoRSCLIP \
  --architecture ViT-L-14 \
  --neighbors 3 5 10 \
  --strengths 0.25 0.50 0.75 1.0
```

Use one global parameter set. Do not tune parameters separately for each dataset.

### 7.5 Cross-backbone validation

After freezing the graph configuration, first evaluate the four ViT-L/14 backbones:

```bash
python scripts/run_all_standard.py \
  --config configs/full_matrix.yaml \
  --stage evaluate \
  --models CLIP RemoteCLIP GeoRSCLIP SkyCLIP50 \
  --architectures ViT-L-14 \
  --methods rs_transclip textgraph_transclip
```

Run the complete 11-architecture matrix only if the focused experiments remain stable.

## 8. Configuration

The main graph settings are:

```yaml
graph:
  k: 3
  mutual: false
  kernel: cosine

text_graph:
  semantic_strength: 1.0
  semantic_power: 1.0
  confidence_power: 1.0
```

- `semantic_strength=0` exactly disables text guidance;
- `confidence_power=0` removes uncertainty fallback;
- `mutual=true` retains only reciprocal visual neighbor edges.

## 9. Diagnostics and paper figures

Set:

```yaml
runtime:
  save_diagnostics: true
```

Solver bundles are stored under:

```text
outputs/results/textgraph/diagnostics/
```

`analyze_graph_edges.py` reports:

- unweighted graph purity;
- weighted graph purity;
- cross-class edge-weight reduction;
- mean semantic affinity;
- mean text confidence;
- mean edge gate factor.

Ground-truth labels are used only by this offline diagnostic script, never by the inference solver.

## 10. Paper scope

The manuscript is located at:

```text
paper/TextGraph_TransCLIP_Chinese_Draft.md
```

The paper should emphasize:

1. cross-class visual neighbors in remote-sensing embeddings;
2. text posteriors as graph-boundary evidence;
3. confidence-controlled fallback to the original visual graph;
4. the relationship between weighted edge purity and classification gain;
5. difficult-class-pair analysis and failure cases.

Previous RAP/SA-RAP partial-class and long-tail experiments are not part of the TextGraph paper claim and should remain in archived result directories.

## 11. Reproducibility

- Use identical cached embeddings for RS and TextGraph.
- Keep one global graph configuration across datasets.
- Record raw CSV files, YAML configuration, and Git commit SHA.
- Report all degraded datasets and at least one failed class-pair case.
- Confirm the exact AID/RSICB dataset versions and class ordering.
- Do not use labels for graph construction, model selection, or parameter tuning.

## License

Repository code is released under the MIT License. Dataset and pretrained-model licenses remain with their original providers.
