#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from rap_transclip.config import load_config
from rap_transclip.feature_extraction import feature_directory, feature_variant


def safe_name(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(" ", "_")


def prediction_path(
    result_root: Path,
    dataset: str,
    model: str,
    architecture: str,
    variant: str,
    method: str,
) -> Path:
    stem = "__".join(
        safe_name(item)
        for item in (dataset, model, architecture, variant, method)
    )
    return result_root / "predictions" / f"{stem}.pt"


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_root = Path(cfg["paths"]["results"])
    raw_path = result_root / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw = pd.read_csv(raw_path)
    key_columns = [
        "dataset",
        "model",
        "architecture",
        "feature_variant",
        "method",
    ]
    raw = raw.drop_duplicates(key_columns, keep="last")
    subset = raw[
        raw["dataset"].isin(args.datasets)
        & (raw["model"] == args.model)
        & (raw["architecture"] == args.architecture)
    ]
    pilot = subset.pivot_table(
        index="method",
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    pilot["Average"] = pilot.mean(axis=1)
    pilot_output = result_root / "pilot_comparison.csv"
    pilot.round(4).to_csv(pilot_output)
    print("\nPilot Top-1 comparison")
    print(pilot.round(4).to_string())

    variant = feature_variant(cfg)
    group_rows: list[dict] = []
    methods = [
        "global_classname",
        "multicrop_classname",
        "object_context",
    ]
    for dataset in args.datasets:
        feature_dir = feature_directory(
            cfg,
            dataset,
            args.model,
            args.architecture,
        )
        classes = json.loads(
            (feature_dir / "classes.json").read_text(encoding="utf-8")
        )
        groups = list(classes["semantic_groups"])

        bundles = {}
        for method in methods:
            path = prediction_path(
                result_root,
                dataset,
                args.model,
                args.architecture,
                variant,
                method,
            )
            if not path.exists():
                raise FileNotFoundError(path)
            bundles[method] = torch.load(path, map_location="cpu")

        labels = bundles["global_classname"]["labels"].long()
        class_group = torch.tensor(
            [
                {"object": 0, "context": 1, "mixed": 2}.get(group, 2)
                for group in groups
            ],
            dtype=torch.long,
        )
        sample_group = class_group[labels]

        for group_name, group_id in [
            ("object", 0),
            ("context", 1),
            ("mixed", 2),
        ]:
            mask = sample_group == group_id
            if not mask.any():
                continue
            row = {
                "dataset": dataset,
                "semantic_group": group_name,
                "num_samples": int(mask.sum().item()),
            }
            for method in methods:
                predictions = bundles[method]["probabilities"].argmax(dim=1)
                accuracy = (predictions[mask] == labels[mask]).float().mean()
                row[f"{method}_top1"] = round(float(accuracy.item() * 100.0), 4)

            object_weights = bundles["object_context"].get("object_weights")
            if object_weights is not None:
                true_class_weights = object_weights[
                    torch.arange(len(labels)),
                    labels,
                ]
                row["mean_true_class_object_weight"] = round(
                    float(true_class_weights[mask].mean().item()),
                    6,
                )
            group_rows.append(row)

    group_frame = pd.DataFrame(group_rows)
    group_output = result_root / "semantic_group_analysis.csv"
    group_frame.to_csv(group_output, index=False)
    print("\nSemantic-group analysis")
    print(group_frame.to_string(index=False))
    print(f"\nSaved: {pilot_output}")
    print(f"Saved: {group_output}")


if __name__ == "__main__":
    main()
