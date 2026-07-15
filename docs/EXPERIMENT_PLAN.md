# Experiment Plan

## Phase 1: correctness

- Reproduce zero-shot and RS-TransCLIP accuracy on EuroSAT, AID, and RESISC45.
- Confirm that class count comes from text prototypes.
- Confirm that labels are only used by metrics.
- Freeze one YAML configuration.

## Phase 2: ten-dataset screening

Run GeoRSCLIP ViT-L/14 on all ten datasets for:

1. zero-shot;
2. RS-TransCLIP;
3. prompt reliability only;
4. prompt reliability plus active prior;
5. complete RAP-TransCLIP.

Decision rule: continue to the complete paper only if the method improves the mean and does not cause large regressions on more than two datasets.

## Phase 3: generality

Add RemoteCLIP and SkyCLIP50 ViT-L/14, then CLIP ViT-L/14. Avoid a full backbone grid unless the main method is stable.

## Phase 4: realistic protocols

Use partial-class and Dirichlet long-tail protocols to test the active prior under its intended setting.

## Required artifacts

- raw_results.csv
- summary.csv
- saved YAML files
- environment lock or pip freeze
- dataset manifest and hashes
- plots of prompt weights and priors
- runtime and memory logs
