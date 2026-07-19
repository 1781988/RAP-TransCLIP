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


def _standardize_per_view(
    scores: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    mean = scores.mean(dim=2, keepdim=True)
    std = scores.std(dim=2, keepdim=True, unbiased=False).clamp_min(eps)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate class-specific local object/structure evidence.

    Every local cue must be supported by several high-response views rather than
    one hard maximum. The returned tensors are class-level object scores and
    per-view class scores used by the consensus term.
    """
    local_features = l2_normalize(local_features)
    object_texts = l2_normalize(object_texts)
    object_mask = object_mask.bool()

    outputs: list[torch.Tensor] = []
    view_outputs: list[torch.Tensor] = []
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

        view_outputs.append(masked.max(dim=3).values)
        per_cue = masked.topk(k=cue_view_count, dim=1).values.mean(dim=1)

        class_scores = []
        for class_id in range(class_count):
            count = min(max_requested, int(valid_counts[class_id].item()))
            values = per_cue[:, class_id, :].topk(k=count, dim=1).values
            class_scores.append(values.mean(dim=1))
        outputs.append(torch.stack(class_scores, dim=1))

    return torch.cat(outputs, dim=0), torch.cat(view_outputs, dim=0)


def class_view_consensus(
    view_class_scores: torch.Tensor,
    view_topk: int,
    center: float,
    temperature: float,
) -> torch.Tensor:
    """Estimate class-specific support consistency across local views."""
    if view_class_scores.ndim != 3:
        raise ValueError("view_class_scores must have shape [N, V, K]")
    standardized = _standardize_per_view(view_class_scores)
    support = torch.sigmoid(
        (standardized - float(center))
        / max(float(temperature), 1e-6)
    )
    count = min(max(1, int(view_topk)), support.shape[1])
    return support.topk(k=count, dim=1).values.mean(dim=1)


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


def _context_candidate_mask(
    context_scores_z: torch.Tensor,
    candidate_topk: int,
) -> torch.Tensor:
    """Return a binary mask for classes already supported by global context."""
    class_count = context_scores_z.shape[1]
    count = int(candidate_topk)
    if count <= 0 or count >= class_count:
        return torch.ones_like(context_scores_z)
    indices = context_scores_z.topk(k=count, dim=1).indices
    mask = torch.zeros_like(context_scores_z)
    mask.scatter_(1, indices, 1.0)
    return mask


def context_anchored_object_fusion(
    context: torch.Tensor,
    objects: torch.Tensor,
    class_consensus: torch.Tensor,
    cfg: dict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    dict[str, torch.Tensor],
]:
    """Add selective local evidence on top of an unchanged context anchor.

    This is the single core inference rule of ObjectContext-CLIP. The context
    branch remains the base classifier. Local evidence can only correct classes
    inside the context Top-M candidate set. By default only positive object
    advantage is added, so local crops cannot directly suppress a context score.
    """
    context_z = _standardize(context)
    object_z = _standardize(objects)
    if class_consensus.shape != object_z.shape:
        raise ValueError("class_consensus must match score shape [N, K]")

    candidate_mask = _context_candidate_mask(
        context_z,
        int(cfg.get("context_candidate_topk", 5)),
    )
    class_gate = torch.sigmoid(
        (object_z - float(cfg.get("object_gate_center", 0.0)))
        / max(float(cfg.get("object_gate_temperature", 0.5)), 1e-6)
    )
    consensus_power = max(
        float(cfg.get("class_consensus_power", 1.0)),
        0.0,
    )
    support = class_gate * class_consensus.clamp_min(1e-4).pow(
        consensus_power
    )

    residual = object_z - context_z
    if bool(cfg.get("positive_residual_only", True)):
        residual = residual.clamp_min(0.0)

    residual_weight = max(float(cfg.get("residual_weight", 0.5)), 0.0)
    correction = residual_weight * candidate_mask * support * residual
    max_boost = float(cfg.get("max_residual_boost", 1.0))
    if max_boost > 0:
        correction = correction.clamp(max=max_boost)

    scores = context_z + correction
    effective_weight = residual_weight * candidate_mask * support
    diagnostics = {
        "mean_candidate_fraction": float(candidate_mask.mean().item()),
        "mean_class_consensus": float(class_consensus.mean().item()),
        "mean_class_gate": float(class_gate.mean().item()),
        "mean_object_support": float(support.mean().item()),
        "mean_object_weight": float(effective_weight.mean().item()),
        "mean_positive_correction": float(correction.mean().item()),
        "max_positive_correction": float(correction.max().item()),
    }
    artifacts = {
        "context_scores": context_z,
        "object_scores": object_z,
        "candidate_mask": candidate_mask,
        "class_gate": class_gate,
        "class_consensus": class_consensus,
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

    objects, per_view = object_evidence_scores(
        local_features,
        object_texts,
        object_mask,
        int(inference_cfg.get("object_topk", 2)),
        int(inference_cfg.get("object_view_topk", 2)),
        int(inference_cfg.get("score_chunk_size", 256)),
    )
    class_consensus = class_view_consensus(
        per_view,
        int(inference_cfg.get("class_consensus_view_topk", 2)),
        float(inference_cfg.get("class_consensus_center", 0.0)),
        float(inference_cfg.get("class_consensus_temperature", 0.5)),
    )
    common_artifacts = {
        "object_scores": _standardize(objects),
        "class_consensus": class_consensus,
    }

    if method == "object_only":
        scores = _standardize(objects)
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 1.0)),
            ),
            scores=scores,
            object_weights=torch.ones_like(scores),
            diagnostics={
                "branch": "object_only",
                "mean_class_consensus": float(class_consensus.mean().item()),
            },
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
                "mean_class_consensus": float(class_consensus.mean().item()),
                "mean_object_weight": weight,
            },
            artifacts=common_artifacts,
        )

    if method == "object_context":
        scores, weights, diagnostics, artifacts = (
            context_anchored_object_fusion(
                context,
                objects,
                class_consensus,
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
