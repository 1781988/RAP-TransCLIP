from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .utils import l2_normalize


@dataclass
class InferenceOutput:
    probabilities: torch.Tensor
    scores: torch.Tensor
    object_weights: torch.Tensor | None
    diagnostics: dict[str, Any]
    artifacts: dict[str, torch.Tensor] = field(default_factory=dict)


def _standardize(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (scores - mean) / std


def _softmax_probabilities(
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.softmax(scores / max(float(temperature), 1e-6), dim=1)


def context_scores(
    global_features: torch.Tensor,
    class_texts: torch.Tensor,
    context_texts: torch.Tensor,
    class_name_weight: float,
) -> torch.Tensor:
    global_features = l2_normalize(global_features)
    class_texts = l2_normalize(class_texts)
    context_texts = l2_normalize(context_texts)
    name_scores = global_features @ class_texts.T
    semantic_scores = global_features @ context_texts.T
    weight = float(min(max(class_name_weight, 0.0), 1.0))
    return weight * name_scores + (1.0 - weight) * semantic_scores


def multicrop_class_scores(
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    class_texts: torch.Tensor,
    local_topk: int,
    local_weight: float,
) -> torch.Tensor:
    global_scores = l2_normalize(global_features) @ l2_normalize(class_texts).T
    local_scores = torch.einsum(
        "nvd,kd->nvk",
        l2_normalize(local_features),
        l2_normalize(class_texts),
    )
    count = min(max(1, int(local_topk)), local_scores.shape[1])
    pooled = local_scores.topk(k=count, dim=1).values.mean(dim=1)
    weight = float(min(max(local_weight, 0.0), 1.0))
    return (1.0 - weight) * global_scores + weight * pooled


def object_evidence_scores(
    local_features: torch.Tensor,
    object_texts: torch.Tensor,
    object_mask: torch.Tensor,
    object_topk: int,
    object_view_topk: int = 2,
    image_chunk_size: int = 256,
) -> torch.Tensor:
    """Aggregate class-specific local object and structure evidence.

    Each local cue is pooled over the strongest ``object_view_topk`` views. The
    strongest ``object_topk`` valid cues are then averaged for each class. This
    keeps the two components that were supported by the completed experiments:
    multi-view evidence aggregation and multiple class-specific local cues.
    """
    local_features = l2_normalize(local_features)
    object_texts = l2_normalize(object_texts)
    object_mask = object_mask.bool()

    outputs: list[torch.Tensor] = []
    class_count = object_texts.shape[0]
    valid_counts = object_mask.sum(dim=1)
    if (valid_counts == 0).any():
        raise ValueError("Every class must contain at least one local cue")

    cue_view_count = min(
        max(1, int(object_view_topk)),
        local_features.shape[1],
    )
    max_requested = max(1, int(object_topk))

    for start in range(0, local_features.shape[0], image_chunk_size):
        end = min(start + image_chunk_size, local_features.shape[0])
        similarities = torch.einsum(
            "bvd,kod->bvko",
            local_features[start:end],
            object_texts,
        )
        expanded_mask = object_mask[None, None, :, :]
        masked = similarities.masked_fill(~expanded_mask, -1e4)
        per_cue = masked.topk(k=cue_view_count, dim=1).values.mean(dim=1)

        class_scores = []
        for class_id in range(class_count):
            count = min(max_requested, int(valid_counts[class_id].item()))
            values = per_cue[:, class_id, :].topk(k=count, dim=1).values
            class_scores.append(values.mean(dim=1))
        outputs.append(torch.stack(class_scores, dim=1))

    return torch.cat(outputs, dim=0)


def fixed_fusion(
    context: torch.Tensor,
    objects: torch.Tensor,
    object_weight: float,
) -> torch.Tensor:
    """Conventional replacement-style fixed fusion used only as a control."""
    context_z = _standardize(context)
    object_z = _standardize(objects)
    weight = float(min(max(object_weight, 0.0), 1.0))
    return (1.0 - weight) * context_z + weight * object_z


def context_anchored_object_residual_fusion(
    context: torch.Tensor,
    objects: torch.Tensor,
    cfg: dict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    dict[str, torch.Tensor],
]:
    """Add class-specific local residuals to a global context anchor.

    The final method intentionally excludes the previously ineffective context
    entropy gate, Top-M candidate mask, and class-consensus weighting. The global
    context branch remains the base classifier. By default, only positive local
    advantages are added, while the object-score gate controls correction size.
    """
    context_z = _standardize(context)
    object_z = _standardize(objects)

    if bool(cfg.get("use_object_gate", True)):
        object_gate = torch.sigmoid(
            (object_z - float(cfg.get("object_gate_center", 0.0)))
            / max(float(cfg.get("object_gate_temperature", 0.5)), 1e-6)
        )
    else:
        object_gate = torch.ones_like(object_z)

    residual = object_z - context_z
    if bool(cfg.get("positive_residual_only", True)):
        residual = residual.clamp_min(0.0)

    residual_weight = max(float(cfg.get("residual_weight", 0.5)), 0.0)
    effective_weight = residual_weight * object_gate
    correction = effective_weight * residual

    max_boost = float(cfg.get("max_residual_boost", 1.0))
    if max_boost > 0:
        correction = correction.clamp(min=-max_boost, max=max_boost)

    scores = context_z + correction
    diagnostics = {
        "mean_object_gate": float(object_gate.mean().item()),
        "mean_object_weight": float(effective_weight.mean().item()),
        "mean_object_correction": float(correction.mean().item()),
        "mean_absolute_object_correction": float(correction.abs().mean().item()),
        "max_absolute_object_correction": float(correction.abs().max().item()),
    }
    artifacts = {
        "context_scores": context_z,
        "object_scores": object_z,
        "object_gate": object_gate,
        "object_residual": residual,
        "object_correction": correction,
    }
    return scores, effective_weight, diagnostics, artifacts


def run_object_context_inference(
    method: str,
    global_features: torch.Tensor,
    local_features: torch.Tensor,
    class_texts: torch.Tensor,
    context_texts: torch.Tensor,
    object_texts: torch.Tensor,
    object_mask: torch.Tensor,
    cfg: dict,
) -> InferenceOutput:
    inference_cfg = cfg["inference"]
    context = context_scores(
        global_features,
        class_texts,
        context_texts,
        float(inference_cfg.get("class_name_weight", 0.5)),
    )

    if method == "global_classname":
        scores = l2_normalize(global_features) @ l2_normalize(class_texts).T
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("global_temperature", 0.01)),
            ),
            scores=scores,
            object_weights=None,
            diagnostics={"branch": "global_classname"},
        )

    if method == "global_context":
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                context,
                float(inference_cfg.get("global_temperature", 0.01)),
            ),
            scores=context,
            object_weights=None,
            diagnostics={"branch": "global_context"},
        )

    if method == "multicrop_classname":
        scores = multicrop_class_scores(
            global_features,
            local_features,
            class_texts,
            int(inference_cfg.get("local_topk", 2)),
            float(inference_cfg.get("multicrop_local_weight", 0.5)),
        )
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("global_temperature", 0.01)),
            ),
            scores=scores,
            object_weights=None,
            diagnostics={"branch": "multicrop_classname"},
        )

    objects = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        int(inference_cfg.get("object_topk", 2)),
        int(inference_cfg.get("object_view_topk", 2)),
        int(inference_cfg.get("score_chunk_size", 256)),
    )
    common_artifacts = {"object_scores": _standardize(objects)}

    if method == "object_only":
        scores = _standardize(objects)
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 1.0)),
            ),
            scores=scores,
            object_weights=torch.ones_like(scores),
            diagnostics={"branch": "object_only"},
            artifacts=common_artifacts,
        )

    if method == "fixed_object_context":
        weight = float(inference_cfg.get("fixed_object_weight", 0.5))
        scores = fixed_fusion(context, objects, weight)
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 1.0)),
            ),
            scores=scores,
            object_weights=torch.full_like(scores, weight),
            diagnostics={
                "branch": "fixed_object_context",
                "mean_object_weight": weight,
            },
            artifacts=common_artifacts,
        )

    if method == "object_context":
        scores, weights, diagnostics, artifacts = (
            context_anchored_object_residual_fusion(
                context,
                objects,
                inference_cfg,
            )
        )
        diagnostics["branch"] = "object_context"
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 1.0)),
            ),
            scores=scores,
            object_weights=weights,
            diagnostics=diagnostics,
            artifacts=artifacts,
        )

    raise ValueError(f"Unsupported method: {method}")
