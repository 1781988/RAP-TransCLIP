#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import apply_overrides, load_config
from rap_transclip.runner import run_experiment


PROTOCOLS = [
    ("partial_25", "partial_class", {"class_fraction": 0.25}),
    ("long_tail_01", "dirichlet_long_tail", {"alpha": 0.1}),
]

VARIANTS = [
    ("A0_rs", "rs_transclip", []),
    (
        "A1_prior_only",
        "rap_transclip",
        [
            "solver.use_prompt_reliability=false",
            "solver.use_active_prior=true",
            "graph.reliability_weighting=false",
        ],
    ),
    (
        "A2_prompt_only",
        "rap_transclip",
        [
            "solver.use_prompt_reliability=true",
            "solver.use_active_prior=false",
            "graph.reliability_weighting=false",
        ],
    ),
    (
        "A3_graph_only",
        "rap_transclip",
        [
            "solver.use_prompt_reliability=false",
            "solver.use_active_prior=false",
            "graph.reliability_weighting=true",
        ],
    ),
    (
        "A4_prompt_prior",
        "rap_transclip",
        [
            "solver.use_prompt_reliability=true",
            "solver.use_active_prior=true",
            "graph.reliability_weighting=false",
        ],
    ),
    (
        "A5_prior_graph",
        "rap_transclip",
        [
            "solver.use_prompt_reliability=false",
            "solver.use_active_prior=true",
            "graph.reliability_weighting=true",
        ],
    ),
    ("A6_full_rap", "rap_transclip", []),
    ("A7_shift_aware", "sa_rap_transclip", []),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or list(base["datasets"])

    for seed in args.seeds:
        for protocol_tag, protocol, protocol_args in PROTOCOLS:
            for variant_tag, method, overrides in VARIANTS:
                cfg = apply_overrides(copy.deepcopy(base), overrides)
                cfg["project"]["seed"] = seed
                cfg["protocol"]["seed"] = seed
                cfg.setdefault("runtime", {})["experiment_tag"] = (
                    f"{protocol_tag}_{variant_tag}_seed{seed}"
                )
                for dataset in datasets:
                    run_experiment(
                        cfg,
                        dataset,
                        args.model,
                        args.architecture,
                        method,
                        protocol_name=protocol,
                        protocol_args=protocol_args,
                    )


if __name__ == "__main__":
    main()
