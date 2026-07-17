from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .utils import l2_normalize


@dataclass
class PromptReliabilityResult:
    weights: torch.Tensor
    prototypes: torch.Tensor
    scores: torch.Tensor


def normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return entropy normalized to [0, 1] along the last dimension."""
    k = probabilities.shape[-1]
    p = probabilities.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)
    return entropy / max(math.log(float(k)), eps)


def _prompt_probabilities(
    image_features: torch.Tensor,
    text_chunk: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    logits = logit_scale * torch.einsum("nd,ckd->cnk", image_features, text_chunk)
    return F.softmax(logits, dim=-1)


def _sparsify_prompt_weights(weights: torch.Tensor, top_prompts_per_class: int) -> torch.Tensor:
    if top_prompts_per_class <= 0 or top_prompts_per_class >= weights.shape[0]:
        return weights
    values, indices = weights.topk(top_prompts_per_class, dim=0)
    sparse = torch.zeros_like(weights)
    sparse.scatter_(0, indices, values)
    return sparse / sparse.sum(dim=0, keepdim=True).clamp_min(1e-8)


def estimate_prompt_reliability(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    cfg: dict,
    logit_scale: float = 100.0,
    target_assignments: torch.Tensor | None = None,
    previous_weights: torch.Tensor | None = None,
) -> PromptReliabilityResult:
    """
    Estimate class-specific prompt weights without materializing [M, N, K].

    Prompt probabilities are evaluated in chunks and discarded after their
    sufficient statistics have been accumulated. Peak memory is therefore
    O(CNK) instead of O(MNK), where C is prompt_chunk_size.
    """
    m, k, _ = text_features.shape
    n = image_features.shape[0]
    chunk_size = max(1, int(cfg.get("prompt_chunk_size", 4)))
    top_count = min(
        max(
            int(cfg.get("min_top_samples", 8)),
            int(n * float(cfg.get("top_fraction", 0.02))),
        ),
        n,
    )
    scores = torch.empty((m, k), device=image_features.device, dtype=image_features.dtype)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        probabilities = _prompt_probabilities(
            image_features,
            text_features[start:end],
            logit_scale,
        )
        for local_prompt in range(end - start):
            prompt_id = start + local_prompt
            p = probabilities[local_prompt]
            entropy = normalized_entropy(p)
            for class_id in range(k):
                class_prob = p[:, class_id]
                values, indices = class_prob.topk(top_count)
                confidence = values.mean()
                selected_entropy = entropy[indices].mean()
                competitors = p[indices].clone()
                competitors[:, class_id] = -1.0
                margin = (values - competitors.max(dim=1).values).mean()
                if target_assignments is None:
                    agreement = confidence
                else:
                    agreement = 1.0 - (
                        values - target_assignments[indices, class_id]
                    ).abs().mean()
                scores[prompt_id, class_id] = (
                    float(cfg.get("confidence_weight", 1.0)) * confidence
                    - float(cfg.get("entropy_weight", 0.6)) * selected_entropy
                    + float(cfg.get("margin_weight", 0.7)) * margin
                    + float(cfg.get("agreement_weight", 0.8)) * agreement
                )

    weights = F.softmax(
        scores / max(float(cfg.get("temperature", 0.15)), 1e-4),
        dim=0,
    )
    weights = _sparsify_prompt_weights(
        weights,
        int(cfg.get("top_prompts_per_class", 0)),
    )
    if previous_weights is not None:
        ema = float(cfg.get("weight_ema", 0.5))
        weights = ema * previous_weights + (1.0 - ema) * weights
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)

    prototypes = l2_normalize(torch.einsum("mk,mkd->kd", weights, text_features))
    return PromptReliabilityResult(weights, prototypes, scores)


def estimate_sample_reliability(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    ensemble_probabilities: torch.Tensor,
    cfg: dict,
    logit_scale: float = 100.0,
) -> torch.Tensor:
    """
    Estimate sample reliability from ensemble entropy and prompt disagreement.

    The prompt-wise variance is computed with running first and second moments,
    so all prompt predictions never need to coexist in GPU memory.
    """
    m = text_features.shape[0]
    chunk_size = max(1, int(cfg.get("prompt_chunk_size", 4)))
    predicted_class = ensemble_probabilities.argmax(dim=1)
    selected_sum = torch.zeros(
        image_features.shape[0],
        device=image_features.device,
        dtype=image_features.dtype,
    )
    selected_sq_sum = torch.zeros_like(selected_sum)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        probabilities = _prompt_probabilities(
            image_features,
            text_features[start:end],
            logit_scale,
        )
        gather_index = predicted_class.view(1, -1, 1).expand(end - start, -1, 1)
        selected = probabilities.gather(2, gather_index).squeeze(-1)
        selected_sum += selected.sum(dim=0)
        selected_sq_sum += selected.square().sum(dim=0)

    prompt_mean = selected_sum / float(m)
    disagreement = (selected_sq_sum / float(m) - prompt_mean.square()).clamp_min(0.0)
    disagreement = disagreement / disagreement.max().clamp_min(1e-8)

    entropy_quality = 1.0 - normalized_entropy(ensemble_probabilities)
    agreement_quality = 1.0 - disagreement
    quality = (
        float(cfg.get("entropy_weight", 0.6)) * entropy_quality
        + float(cfg.get("disagreement_weight", 0.4)) * agreement_quality
    )
    return quality.clamp_min(float(cfg.get("minimum", 0.05))).clamp_max(1.0)
