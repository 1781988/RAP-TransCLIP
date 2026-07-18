#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from rap_transclip.config import load_config
from rap_transclip.feature_extraction import feature_directory, feature_variant


METHODS = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "fixed_object_context",
    "object_context",
]


def safe_name(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(" ", "_")


def prediction_path(
    result_root: Path,
    dataset: str,
    model: str,
    architecture: str,
    variant: str,
    method: str,
    experiment_tag: str,
) -> Path:
    items = (
        dataset,
        model,
        architecture,
        variant,
        method,
        experiment_tag,
    )
    stem = "__".join(safe_name(item) for item in items)
    path = result_root / "predictions" / f"{stem}.pt"
    if path.exists():
        return path

    legacy = "__".join(
        safe_name(item)
        for item in (dataset, model, architecture, variant, method)
    )
    legacy_path = result_root / "predictions" / f"{legacy}.pt"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(path)


def _accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predictions == labels).float().mean().item() * 100.0)


def _mean_optional(
    tensor: torch.Tensor | None,
    mask: torch.Tensor,
) -> float | None:
    if tensor is None:
        return None
    return float(tensor[mask].float().mean().item())


def _true_class_values(
    matrix: torch.Tensor | None,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    if matrix is None or matrix.ndim != 2:
        return None
    return matrix[torch.arange(len(labels)), labels]


def _artifacts(bundle: dict) -> dict[str, torch.Tensor]:
    value = bundle.get("artifacts", {})
    return value if isinstance(value, dict) else {}


def _routing_metrics(
    global_pred: torch.Tensor,
    final_pred: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    final_bundle: dict,
) -> dict[str, float | int]:
    global_correct = global_pred == labels
    final_correct = final_pred == labels
    rescue = (~global_correct) & final_correct & mask
    damage = global_correct & (~final_correct) & mask

    artifacts = _artifacts(final_bundle)
    object_weights = _true_class_values(
        final_bundle.get("object_weights"),
        labels,
    )
    true_consensus = _true_class_values(
        artifacts.get("class_consensus"),
        labels,
    )
    true_gate = _true_class_values(
        artifacts.get("class_gate"),
        labels,
    )
    branch_weight = artifacts.get("object_branch_weight")

    result: dict[str, float | int] = {
        "rescue_count": int(rescue.sum().item()),
        "damage_count": int(damage.sum().item()),
        "net_rescue_count": int(rescue.sum().item() - damage.sum().item()),
        "rescue_rate": round(
            float(rescue.sum().item() / max(int(mask.sum().item()), 1) * 100.0),
            4,
        ),
        "damage_rate": round(
            float(damage.sum().item() / max(int(mask.sum().item()), 1) * 100.0),
            4,
        ),
    }
    for name, tensor in [
        ("mean_true_class_object_weight", object_weights),
        ("mean_true_class_consensus", true_consensus),
        ("mean_true_class_gate", true_gate),
        ("mean_object_branch_weight", branch_weight),
    ]:
        value = _mean_optional(tensor, mask)
        if value is not None:
            result[name] = round(value, 6)
    return result


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
    parser.add_argument("--experiment-tag")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result_root = Path(cfg["paths"]["results"])
    raw_path = result_root / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    experiment_tag = args.experiment_tag or str(
        cfg.get("runtime", {}).get(
            "experiment_tag",
            "object_context_refined_v2",
        )
    )
    variant = feature_variant(cfg)

    raw = pd.read_csv(raw_path)
    if "experiment_tag" not in raw.columns:
        raw["experiment_tag"] = "legacy"
    key_columns = [
        "dataset",
        "model",
        "architecture",
        "feature_variant",
        "method",
        "experiment_tag",
    ]
    raw = raw.drop_duplicates(key_columns, keep="last")
    subset = raw[
        raw["dataset"].isin(args.datasets)
        & (raw["model"] == args.model)
        & (raw["architecture"] == args.architecture)
        & (raw["feature_variant"] == variant)
        & (raw["experiment_tag"] == experiment_tag)
    ]
    if subset.empty:
        raise RuntimeError(
            f"No rows found for experiment_tag={experiment_tag!r}"
        )

    pilot = subset.pivot_table(
        index="method",
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    pilot["Average"] = pilot.mean(axis=1)
    pilot_output = result_root / "pilot_comparison_refined.csv"
    pilot.round(4).to_csv(pilot_output)
    print("\nPilot Top-1 comparison")
    print(pilot.round(4).to_string())

    decision_output = None
    if "object_context" in pilot.index:
        decision_rows = []
        target = float(pilot.loc["object_context", "Average"])
        for baseline in [
            "global_classname",
            "multicrop_classname",
            "global_context",
            "fixed_object_context",
        ]:
            if baseline not in pilot.index:
                continue
            value = float(pilot.loc[baseline, "Average"])
            decision_rows.append(
                {
                    "comparison": f"object_context_minus_{baseline}",
                    "object_context_average": round(target, 4),
                    "baseline_average": round(value, 4),
                    "delta": round(target - value, 4),
                }
            )
        decision = pd.DataFrame(decision_rows)
        decision_output = result_root / "pilot_decision_refined.csv"
        decision.to_csv(decision_output, index=False)

    group_rows: list[dict] = []
    class_rows: list[dict] = []

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
        names = list(classes["names"])
        tokens = list(classes["tokens"])

        bundles = {}
        for method in METHODS:
            path = prediction_path(
                result_root,
                dataset,
                args.model,
                args.architecture,
                variant,
                method,
                experiment_tag,
            )
            bundles[method] = torch.load(path, map_location="cpu")

        labels = bundles["global_classname"]["labels"].long()
        predictions = {
            method: bundles[method]["probabilities"].argmax(dim=1)
            for method in METHODS
        }
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
            row: dict = {
                "dataset": dataset,
                "semantic_group": group_name,
                "num_samples": int(mask.sum().item()),
            }
            for method in METHODS:
                row[f"{method}_top1"] = round(
                    _accuracy(predictions[method][mask], labels[mask]),
                    4,
                )
            row.update(
                _routing_metrics(
                    predictions["global_classname"],
                    predictions["object_context"],
                    labels,
                    mask,
                    bundles["object_context"],
                )
            )
            group_rows.append(row)

        for class_id, (token, name, group) in enumerate(
            zip(tokens, names, groups)
        ):
            mask = labels == class_id
            if not mask.any():
                continue
            row = {
                "dataset": dataset,
                "class_id": class_id,
                "class_token": token,
                "class_name": name,
                "semantic_group": group,
                "num_samples": int(mask.sum().item()),
            }
            for method in METHODS:
                row[f"{method}_top1"] = round(
                    _accuracy(predictions[method][mask], labels[mask]),
                    4,
                )
            row["object_context_minus_global"] = round(
                row["object_context_top1"] - row["global_classname_top1"],
                4,
            )
            row["object_context_minus_multicrop"] = round(
                row["object_context_top1"] - row["multicrop_classname_top1"],
                4,
            )
            row["object_context_minus_context"] = round(
                row["object_context_top1"] - row["global_context_top1"],
                4,
            )
            row.update(
                _routing_metrics(
                    predictions["global_classname"],
                    predictions["object_context"],
                    labels,
                    mask,
                    bundles["object_context"],
                )
            )
            class_rows.append(row)

    group_frame = pd.DataFrame(group_rows)
    group_output = result_root / "semantic_group_analysis_refined.csv"
    group_frame.to_csv(group_output, index=False)
    print("\nSemantic-group analysis")
    print(group_frame.to_string(index=False))

    class_frame = pd.DataFrame(class_rows)
    class_output = result_root / "classwise_analysis_refined.csv"
    class_frame.to_csv(class_output, index=False)

    print(f"\nSaved: {pilot_output}")
    if decision_output is not None:
        print(f"Saved: {decision_output}")
    print(f"Saved: {group_output}")
    print(f"Saved: {class_output}")


if __name__ == "__main__":
    main()
