#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import load_config
from rap_transclip.runner import run_experiment


PROTOCOLS = [
    ("full", "full", {}),
    ("partial_25", "partial_class", {"class_fraction": 0.25}),
    ("partial_50", "partial_class", {"class_fraction": 0.50}),
    ("long_tail_01", "dirichlet_long_tail", {"alpha": 0.1}),
    ("long_tail_05", "dirichlet_long_tail", {"alpha": 0.5}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.02, 0.04, 0.08],
    )
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
        for threshold in args.thresholds:
            for temperature in args.temperatures:
                for protocol_tag, protocol, protocol_args in PROTOCOLS:
                    cfg = copy.deepcopy(base)
                    cfg["project"]["seed"] = seed
                    cfg["protocol"]["seed"] = seed
                    cfg["shift_gate"]["threshold"] = threshold
                    cfg["shift_gate"]["temperature"] = temperature
                    cfg.setdefault("runtime", {})["experiment_tag"] = (
                        f"gate_t{threshold:.2f}_"
                        f"temp{temperature:.2f}_"
                        f"{protocol_tag}_seed{seed}"
                    )
                    for dataset in datasets:
                        run_experiment(
                            cfg,
                            dataset,
                            args.model,
                            args.architecture,
                            "sa_rap_transclip",
                            protocol_name=protocol,
                            protocol_args=protocol_args,
                        )


if __name__ == "__main__":
    main()
