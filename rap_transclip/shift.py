from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from .utils import l2_normalize


@dataclass
class ShiftEstimate:
    score: float
    gate: float
    effective_class_ratio: float
    prior_divergence: float
    active_class_ratio: float
    mean_confidence: float
    confidence_guard: float
    prior: torch.Tensor

    def diagnostics(self) -> dict[str, float]:
        payload = asdict(self)
        payload.pop("prior")
        return {key: float(value) for key, value in payload.items()}


def estimate_batch_shift(
    image_features: torch.Tensor,
    uniform_text_prototypes: torch.Tensor,
    cfg: dict,
    logit_scale: float = 100.0,
) -> ShiftEstimate:
    """Estimate class-prior mismatch using only unlabeled target predictions."""
    image_features = l2_normalize(image_features)
    uniform_text_prototypes = l2_normalize(uniform_text_prototypes)
    probabilities = F.softmax(
        logit_scale * image_features @ uniform_text_prototypes.T,
        dim=1,
    )
    _, k = probabilities.shape
    eps = float(cfg.get("eps", 1e-8))

    prior = probabilities.mean(dim=0).clamp_min(eps)
    prior = prior / prior.sum()

    entropy = -(prior * prior.log()).sum()
    effective_class_ratio = torch.exp(entropy) / float(k)

    uniform = torch.full_like(prior, 1.0 / float(k))
    prior_divergence = (
        (prior * (prior / uniform).log()).sum()
        / max(math.log(float(k)), eps)
    ).clamp(0.0, 1.0)

    peak = probabilities.max(dim=0).values
    mass = prior
    evidence_temperature = max(float(cfg.get("evidence_temperature", 0.05)), 1e-4)
    peak_threshold = float(cfg.get("peak_threshold", 0.30))
    mass_threshold = float(cfg.get("mass_ratio_threshold", 0.20)) / float(k)

    peak_gate = torch.sigmoid((peak - peak_threshold) / evidence_temperature)
    mass_gate = torch.sigmoid((mass - mass_threshold) / evidence_temperature)
    active_class_ratio = (peak_gate * mass_gate).mean()

    mean_confidence = probabilities.max(dim=1).values.mean()
    confidence_reference = float(cfg.get("confidence_reference", 0.25))
    confidence_temperature = max(
        float(cfg.get("confidence_temperature", 0.05)),
        1e-4,
    )
    confidence_guard = torch.sigmoid(
        (mean_confidence - confidence_reference) / confidence_temperature
    )

    weights = cfg.get("weights", {})
    w_effective = float(weights.get("effective_classes", 0.25))
    w_divergence = float(weights.get("prior_divergence", 0.25))
    w_inactive = float(weights.get("inactive_support", 0.50))
    weight_sum = max(w_effective + w_divergence + w_inactive, eps)

    raw_score = (
        w_effective * (1.0 - effective_class_ratio)
        + w_divergence * prior_divergence
        + w_inactive * (1.0 - active_class_ratio)
    ) / weight_sum
    score = (confidence_guard * raw_score).clamp(0.0, 1.0)

    threshold = float(cfg.get("threshold", 0.20))
    gate_temperature = max(float(cfg.get("temperature", 0.04)), 1e-4)
    gate = torch.sigmoid((score - threshold) / gate_temperature)

    return ShiftEstimate(
        score=float(score.item()),
        gate=float(gate.item()),
        effective_class_ratio=float(effective_class_ratio.item()),
        prior_divergence=float(prior_divergence.item()),
        active_class_ratio=float(active_class_ratio.item()),
        mean_confidence=float(mean_confidence.item()),
        confidence_guard=float(confidence_guard.item()),
        prior=prior.detach(),
    )
