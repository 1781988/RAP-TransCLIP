#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rap_transclip.config import apply_overrides, load_config
from rap_transclip.feature_extraction import extract_features
from rap_transclip.runner import run_experiment


def selected(values, defaults):
    return values if values else defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument(
        "--stage",
        choices=["features", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "zero_shot",
            "rs_transclip",
            "rap_transclip",
            "sa_rap_transclip",
        ],
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    datasets = selected(args.datasets, list(cfg["datasets"]))
    enabled_models = [
        name
        for name, spec in cfg["models"].items()
        if spec.get("enabled", False)
    ]
    models = selected(args.models, enabled_models)

    for dataset in datasets:
        for model in models:
            available = list(cfg["models"][model]["architectures"].keys())
            architectures = selected(args.architectures, available)
            for architecture in architectures:
                if architecture not in cfg["models"][model]["architectures"]:
                    continue
                print(f"\n=== {dataset} | {model} | {architecture} ===")
                if args.stage in {"features", "all"}:
                    extract_features(
                        cfg,
                        dataset,
                        model,
                        architecture,
                        args.overwrite,
                    )
                if args.stage in {"evaluate", "all"}:
                    for method in args.methods:
                        run_experiment(
                            cfg,
                            dataset,
                            model,
                            architecture,
                            method,
                        )


if __name__ == "__main__":
    main()
