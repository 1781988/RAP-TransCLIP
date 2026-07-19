#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rap_transclip.config import load_config

MAIN_METHODS = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "object_only",
    "fixed_object_context",
    "object_context",
]
PRIMARY_METHODS = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "object_context",
]
ABLATION_TAGS = [
    "anchor_ablation_no_candidate",
    "anchor_ablation_candidate_top3",
    "anchor_ablation_candidate_top10",
    "anchor_ablation_signed_residual",
    "anchor_ablation_no_consensus",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    protocol = cfg["paper_protocol"]
    raw_path = Path(cfg["paths"]["results"]) / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    frame = pd.read_csv(raw_path)
    keys = [
        "dataset",
        "model",
        "architecture",
        "feature_variant",
        "method",
        "experiment_tag",
    ]
    frame = frame.drop_duplicates(keys, keep="last")

    expected: list[dict[str, str]] = []
    model = protocol["primary_model"]
    architecture = protocol["primary_architecture"]
    for dataset in protocol["all_datasets"]:
        for method in MAIN_METHODS:
            expected.append({
                "dataset": dataset,
                "model": model,
                "architecture": architecture,
                "feature_variant": "clean",
                "method": method,
                "experiment_tag": "anchor_main_georsclip",
            })
        for tag in ["anchor_concept_shuffled", "anchor_concept_generic"]:
            expected.append({
                "dataset": dataset,
                "model": model,
                "architecture": architecture,
                "feature_variant": "clean",
                "method": "object_context",
                "experiment_tag": tag,
            })

    for dataset in protocol["development_datasets"]:
        for tag in ABLATION_TAGS:
            expected.append({
                "dataset": dataset,
                "model": model,
                "architecture": architecture,
                "feature_variant": "clean",
                "method": "object_context",
                "experiment_tag": tag,
            })
        for factor in [1, 2, 4, 8]:
            variant = "clean" if factor == 1 else f"downsample_x{factor}"
            for method in PRIMARY_METHODS:
                expected.append({
                    "dataset": dataset,
                    "model": model,
                    "architecture": architecture,
                    "feature_variant": variant,
                    "method": method,
                    "experiment_tag": f"anchor_resolution_x{factor}",
                })
        for backbone in protocol["cross_backbone_models"]:
            for method in PRIMARY_METHODS:
                expected.append({
                    "dataset": dataset,
                    "model": backbone,
                    "architecture": protocol["cross_backbone_architecture"],
                    "feature_variant": "clean",
                    "method": method,
                    "experiment_tag": f"anchor_cross_backbone_{backbone.lower()}",
                })

    missing = []
    for item in expected:
        mask = pd.Series(True, index=frame.index)
        for key, value in item.items():
            mask &= frame[key] == value
        if not mask.any():
            missing.append(item)

    output = Path(cfg["paths"]["results"]) / "missing_paper_experiments.csv"
    pd.DataFrame(missing).to_csv(output, index=False)
    print(f"Expected unique result rows: {len(expected)}")
    print(f"Missing result rows: {len(missing)}")
    print(f"Missing-row report: {output}")
    if missing:
        raise SystemExit(2)
    print("Paper experiment suite is complete.")


if __name__ == "__main__":
    main()
