from __future__ import annotations

from dataclasses import dataclass
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


def _standardize(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (scores - mean) / std


def _top_margin(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[1] == 1:
        return torch.ones(scores.shape[0], device=scores.device, dtype=scores.dtype)
    values = scores.topk(k=2, dim=1).values
    return values[:, 0] - values[:, 1]


def _margin_reliability(
    scores: torch.Tensor,
    center: float,
    temperature: float,
) -> torch.Tensor:
    margin = _top_margin(scores)
    return torch.sigmoid(
        (margin - float(center)) / max(float(temperature), 1e-6)
    )


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
    image_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate local-view evidence over class-specific cue phrases."""
    local_features = l2_normalize(local_features)
    object_texts = l2_normalize(object_texts)
    object_mask = object_mask.bool()

    outputs: list[torch.Tensor] = []
    view_outputs: list[torch.Tensor] = []
    class_count, _, _ = object_texts.shape
    valid_counts = object_mask.sum(dim=1)
    if (valid_counts == 0).any():
        raise ValueError("Every class must contain at least one local cue")

    for start in range(0, local_features.shape[0], image_chunk_size):
        end = min(start + image_chunk_size, local_features.shape[0])
        similarities = torch.einsum(
            "bvd,kod->bvko",
            local_features[start:end],
            object_texts,
        )
        expanded_mask = object_mask[None, None, :, :]
        masked = similarities.masked_fill(~expanded_mask, -1e4)

        view_class = masked.max(dim=3).values
        view_outputs.append(view_class)

        per_cue = masked.max(dim=1).values
        max_requested = max(1, int(object_topk))
        class_scores = []
        for class_id in range(class_count):
            count = min(max_requested, int(valid_counts[class_id].item()))
            values = per_cue[:, class_id, :].topk(k=count, dim=1).values
            class_scores.append(values.mean(dim=1))
        outputs.append(torch.stack(class_scores, dim=1))

    return torch.cat(outputs, dim=0), torch.cat(view_outputs, dim=0)


def view_consensus(
    view_class_scores: torch.Tensor,
    aggregate_object_scores: torch.Tensor,
) -> torch.Tensor:
    predicted = aggregate_object_scores.argmax(dim=1)
    per_view_prediction = view_class_scores.argmax(dim=2)
    return (
        per_view_prediction == predicted[:, None]
    ).float().mean(dim=1)


def fixed_fusion(
    context: torch.Tensor,
    objects: torch.Tensor,
    object_weight: float,
) -> torch.Tensor:
    context_z = _standardize(context)
    object_z = _standardize(objects)
    weight = float(min(max(object_weight, 0.0), 1.0))
    return (1.0 - weight) * context_z + weight * object_z


def adaptive_object_context_fusion(
    context: torch.Tensor,
    objects: torch.Tensor,
    consensus: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    context_z = _standardize(context)
    object_z = _standardize(objects)

    context_rel = _margin_reliability(
        context_z,
        center=float(cfg.get("context_margin_center", 0.10)),
        temperature=float(cfg.get("margin_temperature", 0.10)),
    )
    object_rel = _margin_reliability(
        object_z,
        center=float(cfg.get("object_margin_center", 0.10)),
        temperature=float(cfg.get("margin_temperature", 0.10)),
    )
    consensus_power = max(float(cfg.get("consensus_power", 1.0)), 0.0)
    object_rel = object_rel * consensus.clamp_min(1e-4).pow(consensus_power)

    context_bias = float(cfg.get("context_bias", 0.15))
    reliability_temperature = max(
        float(cfg.get("reliability_temperature", 0.20)),
        1e-6,
    )
    branch_logits = torch.stack(
        [
            (context_rel + 1e-6).log() + context_bias,
            (object_rel + 1e-6).log(),
        ],
        dim=1,
    ) / reliability_temperature
    object_branch_weight = branch_logits.softmax(dim=1)[:, 1]

    class_gate = torch.sigmoid(
        (object_z - float(cfg.get("class_gate_center", 0.0)))
        / max(float(cfg.get("class_gate_temperature", 0.5)), 1e-6)
    )
    weights = object_branch_weight[:, None] * class_gate
    max_weight = float(cfg.get("max_object_weight", 0.85))
    weights = weights.clamp(0.0, max_weight)

    scores = (1.0 - weights) * context_z + weights * object_z
    diagnostics = {
        "mean_context_reliability": float(context_rel.mean().item()),
        "mean_object_reliability": float(object_rel.mean().item()),
        "mean_view_consensus": float(consensus.mean().item()),
        "mean_object_weight": float(weights.mean().item()),
        "max_object_weight": float(weights.max().item()),
    }
    return scores, weights, diagnostics


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
        int(inference_cfg.get("score_chunk_size", 256)),
    )
    consensus = view_consensus(per_view, objects)

    if method == "object_only":
        scores = _standardize(objects)
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 0.2)),
            ),
            scores=scores,
            object_weights=torch.ones_like(scores),
            diagnostics={
                "branch": "object_only",
                "mean_view_consensus": float(consensus.mean().item()),
            },
        )

    if method == "fixed_object_context":
        weight = float(inference_cfg.get("fixed_object_weight", 0.5))
        scores = fixed_fusion(context, objects, weight)
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 0.2)),
            ),
            scores=scores,
            object_weights=torch.full_like(scores, weight),
            diagnostics={
                "branch": "fixed_object_context",
                "mean_view_consensus": float(consensus.mean().item()),
                "mean_object_weight": weight,
            },
        )

    if method == "object_context":
        scores, weights, diagnostics = adaptive_object_context_fusion(
            context,
            objects,
            consensus,
            inference_cfg,
        )
        diagnostics["branch"] = "object_context"
        return InferenceOutput(
            probabilities=_softmax_probabilities(
                scores,
                float(inference_cfg.get("fusion_temperature", 0.2)),
            ),
            scores=scores,
            object_weights=weights,
            diagnostics=diagnostics,
        )

    raise ValueError(f"Unsupported method: {method}")
