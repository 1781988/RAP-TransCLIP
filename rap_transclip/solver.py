from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch
import torch.nn.functional as F

from .graph import build_graph
from .utils import l2_normalize


@dataclass
class SolverOutput:
    assignments: torch.Tensor
    prototypes: torch.Tensor
    prompt_weights: torch.Tensor | None
    class_prior: torch.Tensor
    sample_reliability: torch.Tensor | None
    iterations: int
    elapsed_seconds: float
    diagnostics: dict[str, object] = field(default_factory=dict)


def _initialize_mu(
    features: torch.Tensor,
    assignments: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    sample_count, feature_dim = features.shape
    class_count = assignments.shape[1]
    count = min(max(1, int(topk)), sample_count)
    mu = torch.empty(
        (class_count, feature_dim),
        device=features.device,
        dtype=features.dtype,
    )
    for class_id in range(class_count):
        values, indices = assignments[:, class_id].topk(count)
        weights = values / values.sum().clamp_min(1e-8)
        mu[class_id] = (weights[:, None] * features[indices]).sum(dim=0)
    return l2_normalize(mu)


def _update_mu(
    features: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    return l2_normalize(
        (assignments.T @ features)
        / assignments.sum(dim=0)[:, None].clamp_min(1e-8)
    )


def _update_shared_variance(
    features: torch.Tensor,
    assignments: torch.Tensor,
    mu: torch.Tensor,
    floor: float,
    chunk_size: int = 2048,
) -> torch.Tensor:
    variance = torch.zeros(
        features.shape[1],
        device=features.device,
        dtype=features.dtype,
    )
    total = assignments.sum().clamp_min(1e-8)
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        diff2 = (features[start:end, None, :] - mu[None, :, :]).square()
        variance += torch.einsum(
            "nk,nkd->d",
            assignments[start:end],
            diff2,
        )
    return (variance / total).clamp_min(floor)


def _gaussian_log_likelihood(
    features: torch.Tensor,
    mu: torch.Tensor,
    variance: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    outputs = []
    inverse = variance.reciprocal()
    log_det = variance.log().sum()
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        diff2 = (features[start:end, None, :] - mu[None, :, :]).square()
        outputs.append(
            -0.5
            * (
                torch.einsum("nkd,d->nk", diff2, inverse)
                + log_det
            )
        )
    return torch.cat(outputs)


def zero_shot_assignments(
    image_features: torch.Tensor,
    text_prototypes: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    return F.softmax(
        logit_scale * image_features @ text_prototypes.T,
        dim=1,
    )


def _solve_transclip(
    image_features: torch.Tensor,
    text_prototypes: torch.Tensor,
    cfg: dict,
    use_text_graph: bool,
) -> SolverOutput:
    started = time.perf_counter()
    image_features = l2_normalize(image_features)
    text_prototypes = l2_normalize(text_prototypes)
    y_hat = zero_shot_assignments(
        image_features,
        text_prototypes,
        float(cfg["zero_shot"]["logit_scale"]),
    )

    z = y_hat.clone()
    class_count = z.shape[1]
    prior = torch.full(
        (class_count,),
        1.0 / class_count,
        device=z.device,
        dtype=z.dtype,
    )
    mu = _initialize_mu(
        image_features,
        z,
        int(cfg["solver"]["prototype_topk"]),
    )
    variance = torch.full(
        (image_features.shape[1],),
        1.0 / image_features.shape[1],
        device=z.device,
        dtype=z.dtype,
    )

    graph_cfg = cfg["graph"]
    text_graph_cfg = cfg.get("text_graph", {})
    graph = build_graph(
        image_features,
        k=int(graph_cfg.get("k", 3)),
        backend=str(graph_cfg.get("backend", "auto")),
        mutual=bool(graph_cfg.get("mutual", False)),
        kernel=str(graph_cfg.get("kernel", "cosine")),
        local_scale_rank=int(
            graph_cfg.get(
                "local_scale_rank",
                graph_cfg.get("k", 3),
            )
        ),
        chunk_size=int(graph_cfg.get("chunk_size", 2048)),
        minimum_similarity=float(
            graph_cfg.get("minimum_similarity", 0.0)
        ),
        semantic_probabilities=y_hat if use_text_graph else None,
        semantic_strength=(
            float(text_graph_cfg.get("semantic_strength", 1.0))
            if use_text_graph
            else 0.0
        ),
        semantic_power=float(
            text_graph_cfg.get("semantic_power", 1.0)
        ),
        confidence_power=float(
            text_graph_cfg.get("confidence_power", 1.0)
        ),
    )

    completed = 0
    previous_z = z
    for outer in range(
        int(cfg["solver"].get("outer_iterations", 10))
    ):
        likelihood = _gaussian_log_likelihood(
            image_features,
            mu,
            variance,
        )
        for _ in range(
            int(cfg["solver"].get("assignment_iterations", 5))
        ):
            propagated = torch.sparse.mm(graph.matrix, z)
            logits = (
                likelihood
                / float(
                    cfg["solver"].get(
                        "likelihood_temperature",
                        50.0,
                    )
                )
                + float(cfg["solver"].get("graph_strength", 1.0))
                * propagated
                + float(
                    cfg["solver"].get(
                        "text_anchor_strength",
                        1.0,
                    )
                )
                * y_hat.clamp_min(1e-8).log()
            )
            z = F.softmax(logits, dim=1)

        mu = _update_mu(image_features, z)
        variance = _update_shared_variance(
            image_features,
            z,
            mu,
            float(cfg["solver"].get("covariance_floor", 1e-4)),
        )
        completed = outer + 1
        delta = (z - previous_z).abs().mean().item()
        previous_z = z.clone()
        if delta < float(
            cfg["solver"].get("convergence_tolerance", 1e-5)
        ):
            break

    method_name = (
        "textgraph_transclip"
        if use_text_graph
        else "rs_transclip"
    )
    diagnostics: dict[str, object] = {
        "solver_branch": method_name,
        "graph_backend": graph.backend,
        "graph_edges": graph.num_edges,
        **graph.diagnostics,
    }
    return SolverOutput(
        assignments=z,
        prototypes=mu,
        prompt_weights=None,
        class_prior=prior,
        sample_reliability=None,
        iterations=completed,
        elapsed_seconds=time.perf_counter() - started,
        diagnostics=diagnostics,
    )


def solve_rs_transclip(
    image_features: torch.Tensor,
    uniform_text_prototypes: torch.Tensor,
    cfg: dict,
) -> SolverOutput:
    return _solve_transclip(
        image_features,
        uniform_text_prototypes,
        cfg,
        use_text_graph=False,
    )


def solve_textgraph_transclip(
    image_features: torch.Tensor,
    uniform_text_prototypes: torch.Tensor,
    cfg: dict,
) -> SolverOutput:
    return _solve_transclip(
        image_features,
        uniform_text_prototypes,
        cfg,
        use_text_graph=True,
    )
