#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TAGS = [
    "object_context_refined_v2",
    "view_topk1",
    "view_topk3",
    "no_class_consensus",
    "shuffled_object_concepts",
    "generic_object_concepts",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/results/object_context_refined_v2/raw_results.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/results/object_context_refined_v2/"
            "refinement_comparison.csv"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AID", "PatternNet", "RESISC45"],
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    subset = frame[
        frame["dataset"].isin(args.datasets)
        & (frame["method"] == "object_context")
        & frame["experiment_tag"].isin(TAGS)
    ].copy()
    key = [
        "dataset",
        "model",
        "architecture",
        "feature_variant",
        "method",
        "experiment_tag",
    ]
    subset = subset.drop_duplicates(key, keep="last")
    table = subset.pivot_table(
        index="experiment_tag",
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    table["Average"] = table.mean(axis=1)
    if "object_context_refined_v2" in table.index:
        reference = table.loc["object_context_refined_v2", "Average"]
        table["Delta_vs_refined"] = table["Average"] - reference
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(output)
    print(table.round(4).to_string())
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
