from __future__ import annotations

import copy
from pathlib import Path

import torch

from rap_transclip.config import load_config
from rap_transclip.graph import build_graph
from rap_transclip.solver import (
    solve_rs_transclip,
    solve_textgraph_transclip,
)
from rap_transclip.utils import l2_normalize


def synthetic_problem(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    class_count, feature_dim = 3, 16
    centers = l2_normalize(
        torch.randn(class_count, feature_dim, generator=generator)
    )
    features = []
    for class_id in range(class_count):
        features.append(
            l2_normalize(
                centers[class_id]
                + 0.12
                * torch.randn(
                    12,
                    feature_dim,
                    generator=generator,
                )
            )
        )
    return torch.cat(features), centers


def cpu_config():
    cfg = copy.deepcopy(
        load_config(Path(__file__).parents[1] / "configs" / "standard.yaml")
    )
    cfg["project"]["device"] = "cpu"
    cfg["graph"]["backend"] = "torch"
    cfg["graph"]["k"] = 3
    cfg["graph"]["chunk_size"] = 64
    cfg["solver"]["outer_iterations"] = 2
    cfg["solver"]["assignment_iterations"] = 2
    return cfg


def assert_probabilities(tensor: torch.Tensor):
    assert torch.isfinite(tensor).all()
    assert torch.allclose(
        tensor.sum(dim=1),
        torch.ones(tensor.shape[0]),
        atol=1e-5,
    )


def test_rs_and_textgraph_solvers_smoke():
    images, texts = synthetic_problem()
    rs_output = solve_rs_transclip(images, texts, cpu_config())
    textgraph_output = solve_textgraph_transclip(
        images,
        texts,
        cpu_config(),
    )
    assert rs_output.assignments.shape == (36, 3)
    assert textgraph_output.assignments.shape == (36, 3)
    assert_probabilities(rs_output.assignments)
    assert_probabilities(textgraph_output.assignments)
    assert textgraph_output.diagnostics["mean_gate_factor"] <= 1.0


def test_zero_semantic_strength_matches_rs_solver():
    images, texts = synthetic_problem()
    cfg = cpu_config()
    rs_output = solve_rs_transclip(images, texts, cfg)
    disabled = copy.deepcopy(cfg)
    disabled["text_graph"]["semantic_strength"] = 0.0
    textgraph_output = solve_textgraph_transclip(
        images,
        texts,
        disabled,
    )
    assert torch.allclose(
        rs_output.assignments,
        textgraph_output.assignments,
        atol=1e-6,
    )


def test_confident_semantic_disagreement_reduces_conductance():
    features = l2_normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.05],
                [0.98, 0.10],
                [0.97, 0.15],
            ]
        )
    )
    probabilities = torch.tensor(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.02, 0.98],
            [0.01, 0.99],
        ]
    )
    visual_graph = build_graph(
        features,
        k=3,
        backend="torch",
    )
    text_graph = build_graph(
        features,
        k=3,
        backend="torch",
        semantic_probabilities=probabilities,
        semantic_strength=1.0,
    )
    assert visual_graph.diagnostics["mean_gate_factor"] == 1.0
    assert text_graph.diagnostics["mean_gate_factor"] < 1.0
    assert text_graph.diagnostics["min_gate_factor"] < 0.5


def test_uncertain_text_predictions_fall_back_to_visual_graph():
    generator = torch.Generator().manual_seed(4)
    features = l2_normalize(torch.randn(8, 6, generator=generator))
    probabilities = torch.full((8, 4), 0.25)
    text_graph = build_graph(
        features,
        k=3,
        backend="torch",
        semantic_probabilities=probabilities,
        semantic_strength=1.0,
    )
    assert abs(text_graph.diagnostics["mean_gate_factor"] - 1.0) < 1e-6
