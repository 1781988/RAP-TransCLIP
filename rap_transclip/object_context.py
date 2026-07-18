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


def _top_margin(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[1] == 1:
        return torch.ones(
            scores.shape[0],
            device=scores.device,
            dtype=scores.dtype,
        )
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
    object_view_topk: int = 2,
    image_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate class-specific local-cue evidence.

    Each cue is required to receive support from the strongest ``object_view_topk``
    local views instead of a single hard maximum. This reduces accidental
    responses caused by one crop while preserving sparse object evidence.
    """
    local_features = l2_normalize(local_features)
    object_texts = l2_normalize(object_texts)
    object_mask = object_mask.bool()

    outputs: list[torch.Tensor] = []
    view_outputs: list[torch.Tensor] = []
    class_count, _, _ = object_texts.shape
    valid_counts = object_mask.sum(dim=1)
    if (valid_counts == 0).any():
        raise ValueError("Every class must contain at least one local cue")

    view_count = local_features.shape[1]
    cue_view_count = min(max(1, int(object_view_topk)), view_count)

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

        per_cue = masked.topk(k=cue_view_count, dim=1).values.mean(dim=1)
        class_scores = []
        max_requested = max(1, int(object_topk))
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
    """Estimate class-specific support consistency across local views.

    Scores are standardized across candidate classes inside each local view,
    converted to soft support values, then averaged over the strongest views
    for every class. The result has shape ``[N, K]`` and lies in ``[0, 1]``.
    """
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
    context_z = _standardize(context)
    object_z = _standardize(objects)
    weight = float(min(max(object_weight, 0.0), 1.0))
    return (1.0 - weight) * context_z + weight * object_z


def adaptive_object_context_fusion(
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
    context_z = _standardize(context)
    object_z = _standardize(objects)
    if class_consensus.shape != object_z.shape:
        raise ValueError("class_consensus must match score shape [N, K]")

    context_rel = _margin_reliability(
        context_z,
        center=float(cfg.get("context_margin_center", 0.10)),
        temperature=float(cfg.get("margin_temperature", 0.10)),
    )
    object_margin_rel = _margin_reliability(
        object_z,
        center=float(cfg.get("object_margin_center", 0.10)),
        temperature=float(cfg.get("margin_temperature", 0.10)),
    )

    object_prediction = object_z.argmax(dim=1)
    predicted_consensus = class_consensus.gather(
        1,
        object_prediction[:, None],
    ).squeeze(1)
    consensus_power = max(float(cfg.get("consensus_power", 1.0)), 0.0)
    object_rel = object_margin_rel * predicted_consensus.clamp_min(
        1e-4
    ).pow(consensus_power)

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
    class_consensus_power = max(
        float(cfg.get("class_consensus_power", 1.0)),
        0.0,
    )
    class_support = class_consensus.clamp_min(1e-4).pow(
        class_consensus_power
    )
    weights = object_branch_weight[:, None] * class_gate * class_support
    max_weight = float(cfg.get("max_object_weight", 0.85))
    weights = weights.clamp(0.0, max_weight)

    scores = (1.0 - weights) * context_z + weights * object_z
    diagnostics = {
        "mean_context_reliability": float(context_rel.mean().item()),
        "mean_object_margin_reliability": float(
            object_margin_rel.mean().item()
        ),
        "mean_object_reliability": float(object_rel.mean().item()),
        "mean_predicted_class_consensus": float(
            predicted_consensus.mean().item()
        ),
        "mean_class_consensus": float(class_consensus.mean().item()),
        "mean_object_branch_weight": float(
            object_branch_weight.mean().item()
        ),
        "mean_class_gate": float(class_gate.mean().item()),
        "mean_object_weight": float(weights.mean().item()),
        "max_object_weight": float(weights.max().item()),
    }
    artifacts = {
        "context_scores": context_z,
        "object_scores": object_z,
        "context_reliability": context_rel,
        "object_margin_reliability": object_margin_rel,
        "object_reliability": object_rel,
        "object_branch_weight": object_branch_weight,
        "class_gate": class_gate,
        "class_consensus": class_consensus,
    }
    return scores, weights, diagnostics, artifacts


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
                float(inference_cfg.get("fusion_temperature", 0.2)),
            ),
            scores=scores,
            object_weights=torch.ones_like(scores),
            diagnostics={
                "branch": "object_only",
                "mean_class_consensus": float(
                    class_consensus.mean().item()
                ),
            },
            artifacts=common_artifacts,
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
                "mean_class_consensus": float(
                    class_consensus.mean().item()
                ),
                "mean_object_weight": weight,
            },
            artifacts=common_artifacts,
        )

    if method == "object_context":
        scores, weights, diagnostics, artifacts = (
            adaptive_object_context_fusion(
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
                float(inference_cfg.get("fusion_temperature", 0.2)),
            ),
            scores=scores,
            object_weights=weights,
            diagnostics=diagnostics,
            artifacts=artifacts,
        )

    raise ValueError(f"Unsupported method: {method}")
