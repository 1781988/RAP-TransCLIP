#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import load_config
from rap_transclip.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--neighbors",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20],
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument(
        "--semantic-powers",
        nargs="+",
        type=float,
        default=[1.0],
    )
    parser.add_argument(
        "--confidence-powers",
        nargs="+",
        type=float,
        default=[1.0],
    )
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or list(base["datasets"])

    for k in args.neighbors:
        for strength in args.strengths:
            for semantic_power in args.semantic_powers:
                for confidence_power in args.confidence_powers:
                    cfg = copy.deepcopy(base)
                    cfg["graph"]["k"] = int(k)
                    cfg["text_graph"]["semantic_strength"] = float(strength)
                    cfg["text_graph"]["semantic_power"] = float(semantic_power)
                    cfg["text_graph"]["confidence_power"] = float(confidence_power)
                    cfg.setdefault("runtime", {})["experiment_tag"] = (
                        f"k{k}_lambda{strength:.2f}_"
                        f"beta{semantic_power:.2f}_gamma{confidence_power:.2f}"
                    )
                    for dataset in datasets:
                        run_experiment(
                            cfg,
                            dataset,
                            args.model,
                            args.architecture,
                            "textgraph_transclip",
                        )


if __name__ == "__main__":
    main()
