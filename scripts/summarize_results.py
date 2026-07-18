#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/results/object_context/raw_results.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/results/object_context/summary.csv",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    metrics = [
        column
        for column in [
            "top1",
            "macro_f1",
            "ece",
            "inference_seconds",
            "peak_cuda_memory_mb",
        ]
        if column in frame.columns
    ]
    group = [
        column
        for column in [
            "dataset",
            "model",
            "architecture",
            "feature_variant",
            "method",
        ]
        if column in frame.columns
    ]
    summary = (
        frame.groupby(group, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
