#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import load_config
from rap_transclip.runner import run_experiment


MAIN_METHODS = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "object_only",
    "fixed_object_context",
    "object_context",
]


VARIANTS = [
    (
        "view_topk1",
        {
            "object_view_topk": 1,
            "object_concept_mode": "correct",
            "consensus_power": 1.0,
            "class_consensus_power": 1.0,
        },
    ),
    (
        "view_topk3",
        {
            "object_view_topk": 3,
            "object_concept_mode": "correct",
            "consensus_power": 1.0,
            "class_consensus_power": 1.0,
        },
    ),
    (
        "no_class_consensus",
        {
            "object_view_topk": 2,
            "object_concept_mode": "correct",
            "consensus_power": 0.0,
            "class_consensus_power": 0.0,
        },
    ),
    (
        "shuffled_object_concepts",
        {
            "object_view_topk": 2,
            "object_concept_mode": "shuffled",
            "consensus_power": 1.0,
            "class_consensus_power": 1.0,
        },
    ),
    (
        "generic_object_concepts",
        {
            "object_view_topk": 2,
            "object_concept_mode": "generic",
            "consensus_power": 1.0,
            "class_consensus_power": 1.0,
        },
    ),
]


def _run(
    base: dict,
    datasets: list[str],
    model: str,
    architecture: str,
    tag: str,
    methods: list[str],
    overrides: dict | None = None,
) -> None:
    cfg = copy.deepcopy(base)
    cfg.setdefault("runtime", {})["experiment_tag"] = tag
    for key, value in (overrides or {}).items():
        cfg.setdefault("inference", {})[key] = value
    for dataset in datasets:
        for method in methods:
            run_experiment(
                cfg,
                dataset,
                model,
                architecture,
                method,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AID", "PatternNet", "RESISC45"],
    )
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Run only inference variants and concept controls",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    if not args.skip_main:
        _run(
            base,
            args.datasets,
            args.model,
            args.architecture,
            "object_context_refined_v2",
            MAIN_METHODS,
            {
                "object_view_topk": 2,
                "object_concept_mode": "correct",
                "consensus_power": 1.0,
                "class_consensus_power": 1.0,
            },
        )

    for tag, overrides in VARIANTS:
        _run(
            base,
            args.datasets,
            args.model,
            args.architecture,
            tag,
            ["object_context"],
            overrides,
        )


if __name__ == "__main__":
    main()
