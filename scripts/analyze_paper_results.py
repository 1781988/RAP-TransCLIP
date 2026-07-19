#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, wilcoxon

from rap_transclip.config import load_config
from rap_transclip.feature_extraction import feature_directory

MAIN_TAG = "anchor_main_georsclip"
PRIMARY_METHODS = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "object_context",
]
BASELINES = [
    "global_classname",
    "multicrop_classname",
    "global_context",
    "fixed_object_context",
]
ABLATION_TAGS = [
    MAIN_TAG,
    "anchor_ablation_no_candidate",
    "anchor_ablation_candidate_top3",
    "anchor_ablation_candidate_top10",
    "anchor_ablation_signed_residual",
    "anchor_ablation_no_consensus",
]


def safe_name(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(" ", "_")


def prediction_path(
    root: Path,
    dataset: str,
    model: str,
    architecture: str,
    variant: str,
    method: str,
    tag: str,
) -> Path:
    stem = "__".join(
        safe_name(item)
        for item in (dataset, model, architecture, variant, method, tag)
    )
    return root / "predictions" / f"{stem}.pt"


def drop_latest(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "model",
        "architecture",
        "feature_variant",
        "method",
        "experiment_tag",
    ]
    return frame.drop_duplicates(keys, keep="last")


def pivot_metric(
    frame: pd.DataFrame,
    tag: str,
    model: str,
    architecture: str,
    metric: str,
    methods: Iterable[str],
) -> pd.DataFrame:
    subset = frame[
        (frame["experiment_tag"] == tag)
        & (frame["model"] == model)
        & (frame["architecture"] == architecture)
        & frame["method"].isin(methods)
    ]
    table = subset.pivot_table(
        index="method",
        columns="dataset",
        values=metric,
        aggfunc="last",
    )
    table["Average"] = table.mean(axis=1)
    return table


def bootstrap_delta(
    labels: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, float]:
    difference = (
        (target == labels).astype(np.float32)
        - (baseline == labels).astype(np.float32)
    )
    observed = float(difference.mean() * 100.0)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        indices = rng.integers(0, len(labels), size=(count, len(labels)))
        values.append(difference[indices].mean(axis=1) * 100.0)
    samples = np.concatenate(values)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return observed, float(low), float(high)


def mcnemar_exact(
    labels: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
) -> tuple[int, int, float]:
    target_correct = target == labels
    baseline_correct = baseline == labels
    rescue = int((target_correct & ~baseline_correct).sum())
    damage = int((~target_correct & baseline_correct).sum())
    discordant = rescue + damage
    p_value = (
        float(binomtest(min(rescue, damage), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return rescue, damage, p_value


def save_main_tables(
    frame: pd.DataFrame,
    output: Path,
    protocol: dict,
) -> dict[str, pd.DataFrame]:
    tables = {}
    for metric in ["top1", "macro_f1", "ece"]:
        table = pivot_metric(
            frame,
            MAIN_TAG,
            protocol["primary_model"],
            protocol["primary_architecture"],
            metric,
            PRIMARY_METHODS,
        )
        table.round(4).to_csv(output / f"table_main_{metric}.csv")
        tables[metric] = table
    return tables


def save_significance(
    cfg: dict,
    output: Path,
) -> pd.DataFrame:
    protocol = cfg["paper_protocol"]
    root = Path(cfg["paths"]["results"])
    model = protocol["primary_model"]
    architecture = protocol["primary_architecture"]
    analysis_cfg = cfg.get("analysis", {})
    repetitions = int(analysis_cfg.get("bootstrap_repetitions", 2000))
    seed = int(analysis_cfg.get("bootstrap_seed", 2027))
    confidence = float(analysis_cfg.get("confidence_level", 0.95))

    rows = []
    deltas = {baseline: [] for baseline in BASELINES}
    for dataset_id, dataset in enumerate(protocol["all_datasets"]):
        target_bundle = torch.load(
            prediction_path(
                root,
                dataset,
                model,
                architecture,
                "clean",
                "object_context",
                MAIN_TAG,
            ),
            map_location="cpu",
        )
        labels = target_bundle["labels"].numpy()
        target = target_bundle["probabilities"].argmax(dim=1).numpy()
        for baseline_id, baseline in enumerate(BASELINES):
            base_bundle = torch.load(
                prediction_path(
                    root,
                    dataset,
                    model,
                    architecture,
                    "clean",
                    baseline,
                    MAIN_TAG,
                ),
                map_location="cpu",
            )
            base_pred = base_bundle["probabilities"].argmax(dim=1).numpy()
            observed, low, high = bootstrap_delta(
                labels,
                target,
                base_pred,
                repetitions,
                seed + dataset_id * 100 + baseline_id,
                confidence,
            )
            rescue, damage, p_value = mcnemar_exact(
                labels,
                target,
                base_pred,
            )
            deltas[baseline].append(observed)
            rows.append({
                "dataset": dataset,
                "baseline": baseline,
                "delta_top1": round(observed, 4),
                "bootstrap_ci_low": round(low, 4),
                "bootstrap_ci_high": round(high, 4),
                "rescue_count": rescue,
                "damage_count": damage,
                "net_rescue_count": rescue - damage,
                "mcnemar_exact_p": p_value,
            })

    pd.DataFrame(rows).to_csv(
        output / "table_significance_per_dataset.csv",
        index=False,
    )

    aggregate_rows = []
    for baseline, values in deltas.items():
        array = np.asarray(values, dtype=float)
        try:
            statistic, p_value = wilcoxon(array, alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        aggregate_rows.append({
            "baseline": baseline,
            "mean_dataset_delta": round(float(array.mean()), 4),
            "median_dataset_delta": round(float(np.median(array)), 4),
            "wins": int((array > 0).sum()),
            "ties": int((array == 0).sum()),
            "losses": int((array < 0).sum()),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p": float(p_value),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        output / "table_significance_across_datasets.csv",
        index=False,
    )
    return aggregate


def save_auxiliary_tables(
    frame: pd.DataFrame,
    output: Path,
    protocol: dict,
) -> None:
    model = protocol["primary_model"]
    architecture = protocol["primary_architecture"]
    development = protocol["development_datasets"]
    all_datasets = protocol["all_datasets"]

    ablation = frame[
        (frame["model"] == model)
        & (frame["architecture"] == architecture)
        & (frame["method"] == "object_context")
        & frame["dataset"].isin(development)
        & frame["experiment_tag"].isin(ABLATION_TAGS)
    ].pivot_table(
        index="experiment_tag",
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    ablation["Average"] = ablation.mean(axis=1)
    ablation.round(4).to_csv(output / "table_ablation.csv")

    concept_tags = {
        MAIN_TAG: "correct",
        "anchor_concept_shuffled": "shuffled",
        "anchor_concept_generic": "no_class_specific_object_evidence",
    }
    concept = frame[
        (frame["model"] == model)
        & (frame["architecture"] == architecture)
        & (frame["method"] == "object_context")
        & frame["dataset"].isin(all_datasets)
        & frame["experiment_tag"].isin(concept_tags)
    ].copy()
    concept["concept_control"] = concept["experiment_tag"].map(concept_tags)
    concept_table = concept.pivot_table(
        index="concept_control",
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    concept_table["Average"] = concept_table.mean(axis=1)
    concept_table.round(4).to_csv(output / "table_concept_controls.csv")

    resolution = frame[
        (frame["model"] == model)
        & (frame["architecture"] == architecture)
        & frame["dataset"].isin(development)
        & frame["experiment_tag"].str.startswith("anchor_resolution_x", na=False)
        & frame["method"].isin(PRIMARY_METHODS)
    ].copy()
    resolution["factor"] = resolution["experiment_tag"].str.replace(
        "anchor_resolution_x", "", regex=False
    )
    resolution_table = resolution.pivot_table(
        index=["method", "factor"],
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    resolution_table["Average"] = resolution_table.mean(axis=1)
    resolution_table.round(4).to_csv(output / "table_resolution.csv")

    cross = frame[
        frame["dataset"].isin(development)
        & frame["method"].isin(PRIMARY_METHODS)
        & frame["experiment_tag"].str.startswith(
            "anchor_cross_backbone_", na=False
        )
    ]
    cross_table = cross.pivot_table(
        index=["model", "method"],
        columns="dataset",
        values="top1",
        aggfunc="last",
    )
    cross_table["Average"] = cross_table.mean(axis=1)
    cross_table.round(4).to_csv(output / "table_cross_backbone.csv")


def save_efficiency(frame: pd.DataFrame, cfg: dict, output: Path) -> None:
    protocol = cfg["paper_protocol"]
    main = frame[
        (frame["experiment_tag"] == MAIN_TAG)
        & (frame["model"] == protocol["primary_model"])
        & (frame["architecture"] == protocol["primary_architecture"])
    ]
    rows = []
    for method, part in main.groupby("method"):
        rows.append({
            "method": method,
            "mean_inference_seconds_per_dataset": round(
                float(part["inference_seconds"].mean()), 4
            ),
            "mean_peak_cuda_memory_mb": round(
                float(part["peak_cuda_memory_mb"].mean()), 2
            ),
        })

    total_cache = 0
    extraction_seconds = 0.0
    feature_dirs = 0
    for dataset in protocol["all_datasets"]:
        directory = feature_directory(
            cfg,
            dataset,
            protocol["primary_model"],
            protocol["primary_architecture"],
        )
        if not directory.exists():
            continue
        feature_dirs += 1
        total_cache += sum(
            item.stat().st_size for item in directory.iterdir() if item.is_file()
        )
        metadata = directory / "metadata.json"
        if metadata.exists():
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            extraction_seconds += float(payload.get("elapsed_seconds", 0.0))

    result = pd.DataFrame(rows)
    result["clean_feature_cache_gb"] = round(total_cache / (1024**3), 4)
    result["total_feature_extraction_seconds"] = round(extraction_seconds, 2)
    result["feature_dataset_count"] = feature_dirs
    result.to_csv(output / "table_efficiency.csv", index=False)


def save_summary(
    output: Path,
    top1: pd.DataFrame,
    aggregate: pd.DataFrame,
    datasets: list[str],
) -> None:
    target = top1.loc["object_context", datasets]

    def delta(baseline: str) -> float:
        return float((target - top1.loc[baseline, datasets]).mean())

    context_delta = target - top1.loc["global_context", datasets]
    verdict = "PASS" if (
        delta("global_classname") > 0.0
        and delta("multicrop_classname") > 0.0
        and delta("global_context") > 0.0
        and int((context_delta > 0).sum()) >= 6
        and float(context_delta.min()) > -5.0
    ) else "REVIEW"

    lines = [
        "# ObjectContext-CLIP result summary",
        "",
        f"Paper decision: **{verdict}**",
        "",
        f"- ObjectContext - Global-ClassName: {delta('global_classname'):.4f}",
        f"- ObjectContext - MultiCrop-ClassName: {delta('multicrop_classname'):.4f}",
        f"- ObjectContext - Global-Context: {delta('global_context'):.4f}",
        f"- Wins/ties/losses vs Global-Context: "
        f"{int((context_delta > 0).sum())}/"
        f"{int((context_delta == 0).sum())}/"
        f"{int((context_delta < 0).sum())}",
        f"- Worst dataset delta vs Global-Context: {float(context_delta.min()):.4f}",
        "",
        "## Across-dataset significance",
        "",
        aggregate.to_markdown(index=False),
        "",
        "The main comparison uses only external or controlled baselines. Earlier "
        "in-house exploratory versions are not treated as competing methods.",
    ]
    (output / "paper_results_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["results"])
    raw_path = root / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    frame = drop_latest(pd.read_csv(raw_path))
    tables = save_main_tables(frame, output, cfg["paper_protocol"])
    aggregate = save_significance(cfg, output)
    save_auxiliary_tables(frame, output, cfg["paper_protocol"])
    save_efficiency(frame, cfg, output)
    save_summary(
        output,
        tables["top1"],
        aggregate,
        list(cfg["paper_protocol"]["all_datasets"]),
    )
    print(f"Saved complete paper analysis to: {output}")


if __name__ == "__main__":
    main()
