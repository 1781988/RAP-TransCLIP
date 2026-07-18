#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import load_config
from rap_transclip.feature_extraction import extract_features
from rap_transclip.runner import run_experiment


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
        "--factors",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "global_classname",
            "multicrop_classname",
            "object_context",
        ],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    for factor in args.factors:
        cfg = copy.deepcopy(base)
        cfg["feature_extraction"]["downsample_factor"] = int(factor)
        cfg["feature_extraction"]["variant"] = (
            "clean" if factor == 1 else f"downsample_x{factor}"
        )
        cfg.setdefault("runtime", {})["experiment_tag"] = (
            f"resolution_x{factor}"
        )
        for dataset in args.datasets:
            extract_features(
                cfg,
                dataset,
                args.model,
                args.architecture,
                args.overwrite,
            )
            for method in args.methods:
                run_experiment(
                    cfg,
                    dataset,
                    args.model,
                    args.architecture,
                    method,
                )


if __name__ == "__main__":
    main()
