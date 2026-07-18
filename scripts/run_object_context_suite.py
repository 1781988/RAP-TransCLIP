#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rap_transclip.config import apply_overrides, load_config
from rap_transclip.feature_extraction import extract_features
from rap_transclip.runner import run_experiment


PILOT_METHODS = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "object_only",
    "fixed_object_context",
    "object_context",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AID", "RESISC45", "PatternNet"],
    )
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--stage",
        choices=["features", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    for dataset in args.datasets:
        print(f"\n=== PILOT {dataset} ===")
        if args.stage in {"features", "all"}:
            extract_features(
                cfg,
                dataset,
                args.model,
                args.architecture,
                args.overwrite,
            )
        if args.stage in {"evaluate", "all"}:
            for method in PILOT_METHODS:
                run_experiment(
                    cfg,
                    dataset,
                    args.model,
                    args.architecture,
                    method,
                )


if __name__ == "__main__":
    main()
