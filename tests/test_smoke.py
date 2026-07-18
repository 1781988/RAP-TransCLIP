from __future__ import annotations

import torch

from rap_transclip.multiview import build_view_specs
from rap_transclip.object_context import (
    adaptive_object_context_fusion,
    object_evidence_scores,
    run_object_context_inference,
)
from rap_transclip.utils import l2_normalize


def synthetic_features(seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    samples, views, classes, objects, dim = 12, 4, 3, 3, 16

    class_centers = l2_normalize(
        torch.randn(classes, dim, generator=generator)
    )
    global_features = l2_normalize(
        class_centers.repeat_interleave(samples // classes, dim=0)
        + 0.15 * torch.randn(samples, dim, generator=generator)
    )
    local_features = l2_normalize(
        global_features[:, None, :]
        + 0.18 * torch.randn(samples, views, dim, generator=generator)
    )
    class_texts = l2_normalize(
        class_centers
        + 0.04 * torch.randn(classes, dim, generator=generator)
    )
    context_texts = l2_normalize(
        class_centers
        + 0.05 * torch.randn(classes, dim, generator=generator)
    )
    object_texts = l2_normalize(
        class_centers[:, None, :]
        + 0.08 * torch.randn(classes, objects, dim, generator=generator)
    )
    object_mask = torch.ones(classes, objects, dtype=torch.bool)
    return (
        global_features,
        local_features,
        class_texts,
        context_texts,
        object_texts,
        object_mask,
    )


def test_view_specs_are_deterministic():
    specs = build_view_specs(
        [0.5, 0.75],
        ["center", "top_left"],
    )
    assert [item.name for item in specs] == [
        "s0p5_center",
        "s0p5_top_left",
        "s0p75_center",
        "s0p75_top_left",
    ]


def test_object_scores_have_expected_shape():
    (
        _,
        local_features,
        _,
        _,
        object_texts,
        object_mask,
    ) = synthetic_features()
    scores, per_view = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=2,
        image_chunk_size=5,
    )
    assert scores.shape == (12, 3)
    assert per_view.shape == (12, 4, 3)
    assert torch.isfinite(scores).all()


def test_adaptive_fusion_is_bounded():
    context = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    objects = torch.tensor([[0.2, 2.5, -0.5], [0.1, 0.2, 0.3]])
    consensus = torch.tensor([1.0, 0.25])
    scores, weights, diagnostics = adaptive_object_context_fusion(
        context,
        objects,
        consensus,
        {
            "context_margin_center": 0.1,
            "object_margin_center": 0.1,
            "margin_temperature": 0.1,
            "reliability_temperature": 0.2,
            "context_bias": 0.15,
            "consensus_power": 1.0,
            "class_gate_center": 0.0,
            "class_gate_temperature": 0.5,
            "max_object_weight": 0.85,
        },
    )
    assert scores.shape == context.shape
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 0.85 + 1e-6)
    assert 0 <= diagnostics["mean_object_weight"] <= 0.85


def test_all_inference_probabilities_are_normalized():
    features = synthetic_features()
    cfg = {
        "inference": {
            "class_name_weight": 0.5,
            "local_topk": 2,
            "multicrop_local_weight": 0.5,
            "object_topk": 2,
            "fixed_object_weight": 0.5,
            "score_chunk_size": 5,
            "global_temperature": 0.01,
            "fusion_temperature": 0.2,
            "context_margin_center": 0.1,
            "object_margin_center": 0.1,
            "margin_temperature": 0.1,
            "reliability_temperature": 0.2,
            "context_bias": 0.15,
            "consensus_power": 1.0,
            "class_gate_center": 0.0,
            "class_gate_temperature": 0.5,
            "max_object_weight": 0.85,
        }
    }
    methods = [
        "global_classname",
        "global_context",
        "multicrop_classname",
        "object_only",
        "fixed_object_context",
        "object_context",
    ]
    for method in methods:
        output = run_object_context_inference(method, *features, cfg)
        assert output.probabilities.shape == (12, 3)
        assert torch.isfinite(output.probabilities).all()
        assert torch.allclose(
            output.probabilities.sum(dim=1),
            torch.ones(12),
            atol=1e-5,
        )
