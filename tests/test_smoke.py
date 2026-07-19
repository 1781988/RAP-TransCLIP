from __future__ import annotations

import pytest
import torch

from rap_transclip.multiview import build_view_specs
from rap_transclip.object_context import (
    context_anchored_object_residual_fusion,
    object_evidence_scores,
    run_object_context_inference,
)
from rap_transclip.runner import _select_local_views
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


def inference_config():
    return {
        "class_name_weight": 0.5,
        "local_topk": 2,
        "multicrop_local_weight": 0.5,
        "object_topk": 2,
        "object_view_topk": 2,
        "fixed_object_weight": 0.5,
        "score_chunk_size": 5,
        "global_temperature": 0.01,
        "fusion_temperature": 1.0,
        "positive_residual_only": True,
        "residual_weight": 0.5,
        "use_object_gate": True,
        "object_gate_center": 0.0,
        "object_gate_temperature": 0.5,
        "max_residual_boost": 1.0,
    }


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


def test_cached_local_view_subset_selection():
    local = torch.arange(2 * 4 * 3).view(2, 4, 3)
    selected, indices = _select_local_views(local, [0, 2])
    assert indices == [0, 2]
    assert selected.shape == (2, 2, 3)
    assert torch.equal(selected[:, 0], local[:, 0])
    assert torch.equal(selected[:, 1], local[:, 2])
    with pytest.raises(IndexError):
        _select_local_views(local, [4])


def test_object_scores_have_expected_shape():
    _, local_features, _, _, object_texts, object_mask = synthetic_features()
    scores = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=2,
        object_view_topk=2,
        image_chunk_size=5,
    )
    assert scores.shape == (12, 3)
    assert torch.isfinite(scores).all()


def test_top_view_pooling_suppresses_single_view_outlier():
    local_features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
    )
    object_texts = torch.tensor([[[1.0, 0.0]]])
    object_mask = torch.ones(1, 1, dtype=torch.bool)
    top1 = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=1,
        object_view_topk=1,
    )
    top2 = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=1,
        object_view_topk=2,
    )
    assert top2.item() < top1.item()


def test_multiple_cues_can_change_class_evidence():
    local_features = torch.tensor([[[1.0, 0.0], [0.8, 0.2]]])
    object_texts = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.2]],
            [[0.0, 1.0], [0.2, 0.8]],
        ]
    )
    object_mask = torch.ones(2, 2, dtype=torch.bool)
    top1 = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=1,
        object_view_topk=2,
    )
    top2 = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=2,
        object_view_topk=2,
    )
    assert top1.shape == top2.shape == (1, 2)
    assert not torch.allclose(top1, top2)


def test_positive_residual_never_reduces_context_scores():
    context = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    objects = torch.tensor([[0.2, 2.5, -0.5], [0.1, 0.2, 0.3]])
    scores, weights, diagnostics, artifacts = (
        context_anchored_object_residual_fusion(
            context,
            objects,
            inference_config(),
        )
    )
    assert scores.shape == context.shape
    assert torch.all(scores >= artifacts["context_scores"] - 1e-6)
    assert torch.all(weights >= 0)
    assert diagnostics["mean_absolute_object_correction"] >= 0


def test_object_gate_bounds_effective_weight():
    context = torch.tensor([[1.0, 0.0, -1.0]])
    objects = torch.tensor([[2.0, 0.5, -0.5]])
    cfg = inference_config()
    _, weights, _, artifacts = context_anchored_object_residual_fusion(
        context,
        objects,
        cfg,
    )
    assert torch.all(weights >= 0)
    assert torch.all(weights <= cfg["residual_weight"] + 1e-6)
    assert torch.all(artifacts["object_gate"] >= 0)
    assert torch.all(artifacts["object_gate"] <= 1)


def test_disabling_object_gate_produces_uniform_weights():
    context = torch.tensor([[1.0, 0.0, -1.0]])
    objects = torch.tensor([[2.0, 0.5, -0.5]])
    cfg = inference_config()
    cfg["use_object_gate"] = False
    _, weights, _, artifacts = context_anchored_object_residual_fusion(
        context,
        objects,
        cfg,
    )
    expected = torch.full_like(weights, cfg["residual_weight"])
    assert torch.allclose(weights, expected)
    assert torch.allclose(artifacts["object_gate"], torch.ones_like(weights))


def test_all_inference_probabilities_are_normalized():
    features = synthetic_features()
    cfg = {"inference": inference_config()}
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
