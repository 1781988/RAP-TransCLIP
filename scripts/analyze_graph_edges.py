#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from rap_transclip.config import load_config
from rap_transclip.graph import build_graph
from rap_transclip.solver import zero_shot_assignments
from rap_transclip.utils import l2_normalize


def edge_statistics(matrix: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    matrix = matrix.coalesce().cpu()
    indices = matrix.indices()
    weights = matrix.values().float()
    rows, cols = indices[0], indices[1]
    keep = rows < cols
    rows = rows[keep]
    cols = cols[keep]
    weights = weights[keep]
    if len(weights) == 0:
        return {
            "num_undirected_edges": 0,
            "unweighted_purity": 0.0,
            "weighted_purity": 0.0,
            "cross_class_weight": 0.0,
        }
    same = labels[rows] == labels[cols]
    total_weight = weights.sum().clamp_min(1e-8)
    same_weight = weights[same].sum() if same.any() else torch.tensor(0.0)
    cross_weight = weights[~same].sum() if (~same).any() else torch.tensor(0.0)
    return {
        "num_undirected_edges": int(len(weights)),
        "unweighted_purity": float(same.float().mean().item() * 100.0),
        "weighted_purity": float((same_weight / total_weight).item() * 100.0),
        "cross_class_weight": float(cross_weight.item()),
    }


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model", default="GeoRSCLIP")
    parser.add_argument("--architecture", default="ViT-L-14")
    parser.add_argument(
        "--output",
        default="outputs/results/textgraph/graph_edge_analysis.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = args.datasets or list(cfg["datasets"])
    device_name = cfg["project"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    for dataset in datasets:
        feature_dir = (
            Path(cfg["paths"]["features"])
            / dataset
            / args.model
            / args.architecture
        )
        images = torch.load(feature_dir / "images.pt", map_location="cpu").float().to(device)
        labels = torch.load(feature_dir / "labels.pt", map_location="cpu").long()
        texts = torch.load(feature_dir / "texts_uniform.pt", map_location="cpu").float().to(device)
        images = l2_normalize(images)
        texts = l2_normalize(texts)
        probabilities = zero_shot_assignments(
            images,
            texts,
            float(cfg["zero_shot"]["logit_scale"]),
        )

        graph_cfg = cfg["graph"]
        visual = build_graph(
            images,
            k=int(graph_cfg["k"]),
            backend=graph_cfg["backend"],
            mutual=bool(graph_cfg["mutual"]),
            kernel=graph_cfg["kernel"],
            local_scale_rank=int(graph_cfg.get("local_scale_rank", graph_cfg["k"])),
            chunk_size=int(graph_cfg["chunk_size"]),
            minimum_similarity=float(graph_cfg.get("minimum_similarity", 0.0)),
        )
        text_cfg = cfg["text_graph"]
        guided = build_graph(
            images,
            k=int(graph_cfg["k"]),
            backend=graph_cfg["backend"],
            mutual=bool(graph_cfg["mutual"]),
            kernel=graph_cfg["kernel"],
            local_scale_rank=int(graph_cfg.get("local_scale_rank", graph_cfg["k"])),
            chunk_size=int(graph_cfg["chunk_size"]),
            minimum_similarity=float(graph_cfg.get("minimum_similarity", 0.0)),
            semantic_probabilities=probabilities,
            semantic_strength=float(text_cfg["semantic_strength"]),
            semantic_power=float(text_cfg["semantic_power"]),
            confidence_power=float(text_cfg["confidence_power"]),
        )

        visual_stats = edge_statistics(visual.matrix, labels)
        guided_stats = edge_statistics(guided.matrix, labels)
        reduction = 100.0 * (
            1.0
            - guided_stats["cross_class_weight"]
            / max(visual_stats["cross_class_weight"], 1e-8)
        )
        row = {
            "dataset": dataset,
            "model": args.model,
            "architecture": args.architecture,
            "k": int(graph_cfg["k"]),
            "visual_unweighted_purity": round(visual_stats["unweighted_purity"], 4),
            "visual_weighted_purity": round(visual_stats["weighted_purity"], 4),
            "textgraph_unweighted_purity": round(guided_stats["unweighted_purity"], 4),
            "textgraph_weighted_purity": round(guided_stats["weighted_purity"], 4),
            "cross_class_weight_reduction": round(reduction, 4),
            "mean_gate_factor": round(guided.diagnostics["mean_gate_factor"], 6),
            "mean_node_confidence": round(guided.diagnostics["mean_node_confidence"], 6),
        }
        append_csv(Path(args.output), row)
        print(row)


if __name__ == "__main__":
    main()
