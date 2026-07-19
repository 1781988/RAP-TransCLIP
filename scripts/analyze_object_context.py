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
    stem = "__".join(
        safe_name(item)
        for item in (
            dataset,
            model,
            architecture,
            variant,
            method,
            experiment_tag,
        )
    )
    path = result_root / "predictions" / f"{stem}.pt"
    if path.exists():
        return path
    raise FileNotFoundError(path)


def _accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predictions == labels).float().mean().item() * 100.0)


def _true_class_values(
    matrix: torch.Tensor | None,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    if matrix is None or matrix.ndim != 2:
        return None
    return matrix[torch.arange(len(labels)), labels]


def _mean_optional(
    tensor: torch.Tensor | None,
    mask: torch.Tensor,
) -> float | None:
    if tensor is None:
        return None
    return float(tensor[mask].float().mean().item())


def _artifacts(bundle: dict) -> dict[str, torch.Tensor]:
    value = bundle.get("artifacts", {})
    return value if isinstance(value, dict) else {}


def _routing_metrics(
    context_pred: torch.Tensor,
    final_pred: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    final_bundle: dict,
) -> dict[str, float | int]:
    context_correct = context_pred == labels
    final_correct = final_pred == labels
    rescue = (~context_correct) & final_correct & mask
    damage = context_correct & (~final_correct) & mask

    artifacts = _artifacts(final_bundle)
    true_weight = _true_class_values(
        final_bundle.get("object_weights"),
        labels,
    )
    true_gate = _true_class_values(
        artifacts.get("object_gate"),
        labels,
    )
    true_residual = _true_class_values(
        artifacts.get("object_residual"),
        labels,
    )
    true_correction = _true_class_values(
        artifacts.get("object_correction"),
        labels,
    )

    result: dict[str, float | int] = {
        "rescue_count_vs_context": int(rescue.sum().item()),
        "damage_count_vs_context": int(damage.sum().item()),
        "net_rescue_count_vs_context": int(
            rescue.sum().item() - damage.sum().item()
        ),
        "rescue_rate_vs_context": round(
            float(rescue.sum().item() / max(int(mask.sum().item()), 1) * 100.0),
            4,
        ),
        "damage_rate_vs_context": round(
            float(damage.sum().item() / max(int(mask.sum().item()), 1) * 100.0),
            4,
        ),
    }
    for name, tensor in [
        ("mean_true_class_object_weight", true_weight),
        ("mean_true_class_object_gate", true_gate),
        ("mean_true_class_object_residual", true_residual),
        ("mean_true_class_object_correction", true_correction),
    ]:
        value = _mean_optional(tensor, mask)
        if value is not None:
            result[name] = round(value, 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
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
        cfg.get("runtime", {}).get("experiment_tag", "core_main_georsclip")
    )
    variant = feature_variant(cfg)

    raw = pd.read_csv(raw_path)
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

    comparison = subset.pivot_table(
        index="method",
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    comparison["Average"] = comparison.mean(axis=1)
    comparison_output = result_root / "main_comparison.csv"
    comparison.round(4).to_csv(comparison_output)
    print("\nMain Top-1 comparison")
    print(comparison.round(4).to_string())

    decision_rows = []
    if "object_context" in comparison.index:
        target = float(comparison.loc["object_context", "Average"])
        for baseline in [
            "global_classname",
            "multicrop_classname",
            "global_context",
            "fixed_object_context",
        ]:
            if baseline not in comparison.index:
                continue
            value = float(comparison.loc[baseline, "Average"])
            decision_rows.append({
                "comparison": f"object_context_minus_{baseline}",
                "object_context_average": round(target, 4),
                "baseline_average": round(value, 4),
                "delta": round(target - value, 4),
            })
    decision_output = result_root / "main_decision.csv"
    pd.DataFrame(decision_rows).to_csv(decision_output, index=False)

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

        labels = bundles["global_context"]["labels"].long()
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
            row["object_context_minus_context"] = round(
                row["object_context_top1"] - row["global_context_top1"],
                4,
            )
            row.update(
                _routing_metrics(
                    predictions["global_context"],
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
                row["object_context_top1"]
                - row["multicrop_classname_top1"],
                4,
            )
            row["object_context_minus_context"] = round(
                row["object_context_top1"] - row["global_context_top1"],
                4,
            )
            row.update(
                _routing_metrics(
                    predictions["global_context"],
                    predictions["object_context"],
                    labels,
                    mask,
                    bundles["object_context"],
                )
            )
            class_rows.append(row)

    group_output = result_root / "semantic_group_analysis.csv"
    pd.DataFrame(group_rows).to_csv(group_output, index=False)
    class_output = result_root / "classwise_analysis.csv"
    pd.DataFrame(class_rows).to_csv(class_output, index=False)

    print(f"\nSaved: {comparison_output}")
    print(f"Saved: {decision_output}")
    print(f"Saved: {group_output}")
    print(f"Saved: {class_output}")


if __name__ == "__main__":
    main()
