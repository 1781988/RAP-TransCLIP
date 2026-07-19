#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import pandas as pd

from rap_transclip.config import load_config
from rap_transclip.feature_extraction import extract_features, feature_variant
from rap_transclip.runner import run_experiment

MAIN_TAG = "core_main_georsclip"
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

# Focused component tests for the final lean method. The already rejected
# uncertainty gate, candidate mask, and class-consensus fusion are not rerun.
ABLATIONS = [
    ("core_ablation_no_object_gate", {"use_object_gate": False}),
    ("core_ablation_signed_residual", {"positive_residual_only": False}),
    ("core_ablation_single_view", {"object_view_topk": 1}),
    ("core_ablation_single_cue", {"object_topk": 1}),
]

CONCEPT_CONTROLS = [
    ("core_concept_shuffled", "shuffled"),
    ("core_concept_generic", "generic"),
]


def _copy_with(
    base: dict,
    *,
    tag: str,
    inference: dict | None = None,
    model: str | None = None,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg.setdefault("runtime", {})["experiment_tag"] = tag
    cfg.setdefault("runtime", {})["save_predictions"] = True
    for key, value in (inference or {}).items():
        cfg.setdefault("inference", {})[key] = value
    if model is not None:
        for name in cfg["models"]:
            cfg["models"][name]["enabled"] = name == model
    return cfg


def _result_exists(
    cfg: dict,
    dataset: str,
    model: str,
    architecture: str,
    method: str,
) -> bool:
    path = Path(cfg["paths"]["results"]) / "raw_results.csv"
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    required = {
        "dataset",
        "model",
        "architecture",
        "feature_variant",
        "method",
        "experiment_tag",
    }
    if not required.issubset(frame.columns):
        return False
    tag = str(cfg.get("runtime", {}).get("experiment_tag", ""))
    mask = (
        (frame["dataset"] == dataset)
        & (frame["model"] == model)
        & (frame["architecture"] == architecture)
        & (frame["feature_variant"] == feature_variant(cfg))
        & (frame["method"] == method)
        & (frame["experiment_tag"] == tag)
    )
    return bool(mask.any())


def _run_grid(
    cfg: dict,
    datasets: Iterable[str],
    model: str,
    architecture: str,
    methods: Iterable[str],
    *,
    extract: bool,
    overwrite_features: bool,
    force_evaluate: bool,
) -> None:
    for dataset in datasets:
        print(
            f"\n=== {cfg['runtime']['experiment_tag']} | "
            f"{dataset} | {model} | {architecture} ==="
        )
        if extract:
            extract_features(
                cfg,
                dataset,
                model,
                architecture,
                overwrite=overwrite_features,
            )
        for method in methods:
            if not force_evaluate and _result_exists(
                cfg, dataset, model, architecture, method
            ):
                print(f"Skipping existing result: {dataset} | {method}")
                continue
            run_experiment(cfg, dataset, model, architecture, method)


def _preflight(cfg: dict, include_cross_backbone: bool) -> None:
    protocol = cfg["paper_protocol"]
    missing: list[str] = []
    for dataset in protocol["all_datasets"]:
        index = Path(cfg["paths"]["indexes"]) / f"{dataset}.jsonl"
        if not index.exists():
            missing.append(str(index))

    models = [protocol["primary_model"]]
    if include_cross_backbone:
        models = list(protocol["cross_backbone_models"])
    architecture = protocol["cross_backbone_architecture"]
    for model in models:
        spec = cfg["models"][model]["architectures"][architecture]
        checkpoint = spec.get("checkpoint")
        if checkpoint and not Path(checkpoint).exists():
            missing.append(str(checkpoint))

    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Paper-suite preflight failed. Missing required files:\n" + formatted
        )
    Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    print("Preflight passed.")


def run_main(args, base: dict) -> None:
    protocol = base["paper_protocol"]
    cfg = _copy_with(base, tag=MAIN_TAG)
    _run_grid(
        cfg,
        protocol["all_datasets"],
        protocol["primary_model"],
        protocol["primary_architecture"],
        MAIN_METHODS,
        extract=not args.skip_feature_extraction,
        overwrite_features=args.overwrite_features,
        force_evaluate=args.force_evaluate,
    )


def run_ablation(args, base: dict) -> None:
    protocol = base["paper_protocol"]
    for tag, overrides in ABLATIONS:
        cfg = _copy_with(base, tag=tag, inference=overrides)
        _run_grid(
            cfg,
            protocol["development_datasets"],
            protocol["primary_model"],
            protocol["primary_architecture"],
            ["object_context"],
            extract=False,
            overwrite_features=False,
            force_evaluate=args.force_evaluate,
        )


def run_concepts(args, base: dict) -> None:
    protocol = base["paper_protocol"]
    for tag, mode in CONCEPT_CONTROLS:
        cfg = _copy_with(
            base,
            tag=tag,
            inference={"object_concept_mode": mode},
        )
        _run_grid(
            cfg,
            protocol["all_datasets"],
            protocol["primary_model"],
            protocol["primary_architecture"],
            ["object_context"],
            extract=False,
            overwrite_features=False,
            force_evaluate=args.force_evaluate,
        )


def run_resolution(args, base: dict) -> None:
    protocol = base["paper_protocol"]
    for factor in args.resolution_factors:
        cfg = _copy_with(base, tag=f"core_resolution_x{factor}")
        cfg["feature_extraction"]["downsample_factor"] = int(factor)
        cfg["feature_extraction"]["variant"] = (
            "clean" if factor == 1 else f"downsample_x{factor}"
        )
        _run_grid(
            cfg,
            protocol["development_datasets"],
            protocol["primary_model"],
            protocol["primary_architecture"],
            PRIMARY_METHODS,
            extract=not args.skip_feature_extraction,
            overwrite_features=args.overwrite_features,
            force_evaluate=args.force_evaluate,
        )


def run_cross_backbone(args, base: dict) -> None:
    protocol = base["paper_protocol"]
    architecture = protocol["cross_backbone_architecture"]
    for model in protocol["cross_backbone_models"]:
        cfg = _copy_with(
            base,
            tag=f"core_cross_backbone_{model.lower()}",
            model=model,
        )
        _run_grid(
            cfg,
            protocol["development_datasets"],
            model,
            architecture,
            PRIMARY_METHODS,
            extract=not args.skip_feature_extraction,
            overwrite_features=args.overwrite_features,
            force_evaluate=args.force_evaluate,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete lean ObjectContext-CLIP paper suite."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "preflight",
            "main",
            "ablation",
            "concepts",
            "resolution",
            "cross_backbone",
            "all",
        ],
        default=["all"],
    )
    parser.add_argument(
        "--resolution-factors",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    stages = args.stages
    if "all" in stages:
        stages = [
            "preflight",
            "main",
            "ablation",
            "concepts",
            "resolution",
            "cross_backbone",
        ]

    if "preflight" in stages:
        _preflight(base, include_cross_backbone="cross_backbone" in stages)
    if "main" in stages:
        run_main(args, base)
    if "ablation" in stages:
        run_ablation(args, base)
    if "concepts" in stages:
        run_concepts(args, base)
    if "resolution" in stages:
        run_resolution(args, base)
    if "cross_backbone" in stages:
        run_cross_backbone(args, base)


if __name__ == "__main__":
    main()
