from __future__ import annotations

import pytest
import torch

from rap_transclip.multiview import build_view_specs
from rap_transclip.object_context import (
    class_view_consensus,
    normalized_context_uncertainty,
    object_evidence_scores,
    run_object_context_inference,
    uncertainty_gated_object_residual_fusion,
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
        "fusion_temperature": 1.0,
        "use_uncertainty_gate": True,
        "uncertainty_temperature": 1.0,
        "positive_residual_only": True,
        "residual_weight": 0.5,
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


def test_class_consensus_is_bounded():
    view_scores = torch.tensor(
        [[[3.0, 0.0, -1.0], [2.5, 0.1, -0.5], [-0.2, 2.0, 0.0]]]
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


def test_context_uncertainty_is_higher_for_ambiguous_scores():
    scores = torch.tensor(
        [[5.0, 0.0, -1.0], [0.1, 0.0, -0.1]]
    )
    uncertainty = normalized_context_uncertainty(scores, temperature=1.0)
    assert uncertainty.shape == (2,)
    assert uncertainty[1] > uncertainty[0]
    assert torch.all(uncertainty >= 0)
    assert torch.all(uncertainty <= 1)


def test_positive_residual_never_reduces_context_scores():
    context = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    objects = torch.tensor([[0.2, 2.5, -0.5], [0.1, 0.2, 0.3]])
    consensus = torch.ones_like(context)
    scores, weights, diagnostics, artifacts = (
        uncertainty_gated_object_residual_fusion(
            context,
            objects,
            consensus,
            inference_config(),
        )
    )
    assert scores.shape == context.shape
    assert torch.all(scores >= artifacts["context_scores"] - 1e-6)
    assert torch.all(weights >= 0)
    assert 0 <= diagnostics["mean_context_uncertainty"] <= 1


def test_uncertainty_gate_never_exceeds_ungated_correction():
    context = torch.tensor([[5.0, 0.0, -1.0], [0.1, 0.0, -0.1]])
    objects = torch.tensor([[0.0, 4.0, -1.0], [0.0, 4.0, -1.0]])
    consensus = torch.ones_like(context)
    gated_cfg = inference_config()
    ungated_cfg = dict(gated_cfg)
    ungated_cfg["use_uncertainty_gate"] = False
    _, _, _, gated = uncertainty_gated_object_residual_fusion(
        context,
        objects,
        consensus,
        gated_cfg,
    )
    _, _, _, ungated = uncertainty_gated_object_residual_fusion(
        context,
        objects,
        consensus,
        ungated_cfg,
    )
    assert torch.all(
        gated["object_correction"] <= ungated["object_correction"] + 1e-6
    )
    assert torch.any(
        gated["object_correction"] < ungated["object_correction"] - 1e-6
    )


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
