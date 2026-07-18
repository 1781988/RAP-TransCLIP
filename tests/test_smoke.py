from __future__ import annotations

import pytest
import torch

from rap_transclip.multiview import build_view_specs
from rap_transclip.object_context import (
    adaptive_object_context_fusion,
    class_view_consensus,
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
        "class_consensus_view_topk": 2,
        "class_consensus_center": 0.0,
        "class_consensus_temperature": 0.5,
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
        "class_consensus_power": 1.0,
        "class_gate_center": 0.0,
        "class_gate_temperature": 0.5,
        "max_object_weight": 0.85,
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
        object_view_topk=2,
        image_chunk_size=5,
    )
    assert scores.shape == (12, 3)
    assert per_view.shape == (12, 4, 3)
    assert torch.isfinite(scores).all()


def test_top_view_pooling_suppresses_single_view_outlier():
    local_features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
    )
    object_texts = torch.tensor([[[1.0, 0.0]]])
    object_mask = torch.ones(1, 1, dtype=torch.bool)
    top1, _ = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=1,
        object_view_topk=1,
    )
    top2, _ = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        object_topk=1,
        object_view_topk=2,
    )
    assert top2.item() < top1.item()


def test_class_consensus_is_class_specific_and_bounded():
    view_scores = torch.tensor(
        [
            [
                [3.0, 0.0, -1.0],
                [2.5, 0.1, -0.5],
                [-0.2, 2.0, 0.0],
            ]
        ]
    )
    consensus = class_view_consensus(
        view_scores,
        view_topk=2,
        center=0.0,
        temperature=0.5,
    )
    assert consensus.shape == (1, 3)
    assert torch.all(consensus >= 0)
    assert torch.all(consensus <= 1)
    assert consensus[0, 0] > consensus[0, 2]


def test_adaptive_fusion_is_bounded_and_returns_artifacts():
    context = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    objects = torch.tensor([[0.2, 2.5, -0.5], [0.1, 0.2, 0.3]])
    consensus = torch.tensor(
        [[0.2, 0.9, 0.1], [0.4, 0.5, 0.6]]
    )
    scores, weights, diagnostics, artifacts = (
        adaptive_object_context_fusion(
            context,
            objects,
            consensus,
            inference_config(),
        )
    )
    assert scores.shape == context.shape
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 0.85 + 1e-6)
    assert 0 <= diagnostics["mean_object_weight"] <= 0.85
    assert artifacts["class_consensus"].shape == context.shape
    assert artifacts["object_branch_weight"].shape == (2,)


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
