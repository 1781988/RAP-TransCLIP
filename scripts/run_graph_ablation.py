#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy

from rap_transclip.config import apply_overrides, load_config
from rap_transclip.runner import run_experiment


VARIANTS = [
    ("A0_rs_visual_graph", "rs_transclip", []),
    (
        "A1_visual_semantic_no_fallback",
        "textgraph_transclip",
        [
            "text_graph.semantic_strength=1.0",
            "text_graph.semantic_power=1.0",
            "text_graph.confidence_power=0.0",
        ],
    ),
    (
        "A2_textgraph_full",
        "textgraph_transclip",
        [
            "text_graph.semantic_strength=1.0",
            "text_graph.semantic_power=1.0",
            "text_graph.confidence_power=1.0",
        ],
    ),
    (
        "A3_textgraph_mutual",
        "textgraph_transclip",
        [
            "graph.mutual=true",
            "text_graph.semantic_strength=1.0",
            "text_graph.semantic_power=1.0",
            "text_graph.confidence_power=1.0",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    args = parser.parse_args()

    base = load_config(args.config)
    datasets = args.datasets or list(base["datasets"])

    for tag, method, overrides in VARIANTS:
        cfg = apply_overrides(copy.deepcopy(base), overrides)
        cfg.setdefault("runtime", {})["experiment_tag"] = tag
        for dataset in datasets:
            run_experiment(
                cfg,
                dataset,
                args.model,
                args.architecture,
                method,
            )


if __name__ == "__main__":
    main()
