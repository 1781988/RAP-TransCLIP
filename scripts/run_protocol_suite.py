#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import load_config
from rap_transclip.runner import run_experiment


PROTOCOLS = [
    ("partial_25", "partial_class", {"class_fraction": 0.25}),
    ("partial_50", "partial_class", {"class_fraction": 0.50}),
    ("partial_75", "partial_class", {"class_fraction": 0.75}),
    ("long_tail_01", "dirichlet_long_tail", {"alpha": 0.1}),
    ("long_tail_05", "dirichlet_long_tail", {"alpha": 0.5}),
    ("long_tail_10", "dirichlet_long_tail", {"alpha": 1.0}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "rs_transclip",
            "rap_transclip",
            "sa_rap_transclip",
        ],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument("--severe-only", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or list(base["datasets"])
    protocols = [PROTOCOLS[0], PROTOCOLS[3]] if args.severe_only else PROTOCOLS

    for seed in args.seeds:
        for tag, protocol, protocol_args in protocols:
            cfg = copy.deepcopy(base)
            cfg["project"]["seed"] = seed
            cfg["protocol"]["seed"] = seed
            cfg.setdefault("runtime", {})["experiment_tag"] = f"{tag}_seed{seed}"
            for dataset in datasets:
                for method in args.methods:
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
