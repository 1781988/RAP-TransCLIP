from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class GraphResult:
    matrix: torch.Tensor
    backend: str
    num_edges: int
    diagnostics: dict[str, float] = field(default_factory=dict)


def _faiss_knn(features: np.ndarray, k: int):
    import faiss

    features = features.astype(np.float32, copy=False)
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    similarities, indices = index.search(features, k + 1)
    return similarities[:, 1:], indices[:, 1:]


def _torch_knn(features: torch.Tensor, k: int, chunk_size: int):
    values = []
    indices = []
    n = features.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        similarities = features[start:end] @ features.T
        similarities[
            torch.arange(end - start, device=features.device),
            torch.arange(start, end, device=features.device),
        ] = -float("inf")
        chunk_values, chunk_indices = similarities.topk(k=k, dim=1)
        values.append(chunk_values.cpu())
        indices.append(chunk_indices.cpu())
    return torch.cat(values).numpy(), torch.cat(indices).numpy()


def _normalized_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("semantic probabilities must have shape [N, K]")
    class_count = probabilities.shape[1]
    if class_count <= 1:
        return torch.ones(probabilities.shape[0], device=probabilities.device)
    p = probabilities.clamp_min(1e-8)
    entropy = -(p * p.log()).sum(dim=1) / math.log(float(class_count))
    return (1.0 - entropy).clamp(0.0, 1.0)


def build_graph(
    features: torch.Tensor,
    k: int = 3,
    backend: str = "auto",
    mutual: bool = False,
    kernel: str = "cosine",
    local_scale_rank: int = 3,
    chunk_size: int = 2048,
    minimum_similarity: float = 0.0,
    semantic_probabilities: torch.Tensor | None = None,
    semantic_strength: float = 0.0,
    semantic_power: float = 1.0,
    confidence_power: float = 1.0,
) -> GraphResult:
    """Build a visual kNN graph with optional text-posterior edge conductance.

    The visual kNN topology is shared by RS-TransCLIP and TextGraph-TransCLIP.
    Text guidance only changes the conductance of existing visual edges.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    sample_count = features.shape[0]
    if semantic_probabilities is not None and semantic_probabilities.shape[0] != sample_count:
        raise ValueError("semantic probabilities and features must share N")
    if sample_count <= 1:
        indices = torch.zeros((2, 0), dtype=torch.long, device=features.device)
        values = torch.zeros(0, dtype=features.dtype, device=features.device)
        matrix = torch.sparse_coo_tensor(indices, values, (sample_count, sample_count)).coalesce()
        return GraphResult(matrix, "none", 0, {})

    k = min(max(1, int(k)), sample_count - 1)
    selected_backend = backend
    if backend in {"auto", "faiss"}:
        try:
            similarities_np, indices_np = _faiss_knn(
                features.detach().cpu().numpy(), k
            )
            selected_backend = "faiss"
        except Exception:
            if backend == "faiss":
                raise
            similarities_np, indices_np = _torch_knn(features, k, chunk_size)
            selected_backend = "torch"
    elif backend == "torch":
        similarities_np, indices_np = _torch_knn(features, k, chunk_size)
    else:
        raise ValueError(f"Unsupported graph backend: {backend}")

    similarities_np = np.maximum(similarities_np, minimum_similarity)
    rows = np.repeat(np.arange(sample_count), k)
    cols = indices_np.reshape(-1)

    if kernel == "cosine":
        visual_weights = similarities_np.reshape(-1)
    elif kernel == "rbf":
        rank = min(max(1, local_scale_rank), k) - 1
        local_scale = np.maximum(1.0 - similarities_np[:, rank], 1e-4)
        visual_weights = np.exp(
            -np.maximum(1.0 - similarities_np, 0.0).reshape(-1)
            / np.repeat(local_scale, k)
        )
    else:
        raise ValueError(f"Unsupported graph kernel: {kernel}")

    weights = visual_weights.astype(np.float64, copy=True)
    diagnostics: dict[str, float] = {
        "mean_visual_similarity": float(np.mean(similarities_np)),
    }

    strength = float(np.clip(semantic_strength, 0.0, 1.0))
    if semantic_probabilities is not None and strength > 0.0:
        probabilities = semantic_probabilities.detach().float()
        probabilities = probabilities / probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        sqrt_probabilities = probabilities.clamp_min(1e-8).sqrt()
        row_index = torch.from_numpy(rows).to(probabilities.device)
        col_index = torch.from_numpy(cols).to(probabilities.device)

        semantic_affinity = (
            sqrt_probabilities[row_index] * sqrt_probabilities[col_index]
        ).sum(dim=1).clamp(0.0, 1.0)
        confidence = _normalized_confidence(probabilities)
        pair_confidence = (
            confidence[row_index] * confidence[col_index]
        ).clamp_min(0.0).sqrt()
        pair_confidence = pair_confidence.pow(max(float(confidence_power), 0.0))
        semantic_affinity = semantic_affinity.pow(
            max(float(semantic_power), 1e-6)
        )

        gate_factor = 1.0 - strength * pair_confidence * (
            1.0 - semantic_affinity
        )
        weights *= gate_factor.cpu().numpy().astype(np.float64, copy=False)
        diagnostics.update(
            {
                "mean_semantic_affinity": float(semantic_affinity.mean().item()),
                "mean_node_confidence": float(confidence.mean().item()),
                "mean_gate_factor": float(gate_factor.mean().item()),
                "min_gate_factor": float(gate_factor.min().item()),
            }
        )
    else:
        diagnostics.update(
            {
                "mean_semantic_affinity": 1.0,
                "mean_node_confidence": 0.0,
                "mean_gate_factor": 1.0,
                "min_gate_factor": 1.0,
            }
        )

    directed = sp.csr_matrix(
        (weights, (rows, cols)),
        shape=(sample_count, sample_count),
    )
    graph = directed.minimum(directed.T) if mutual else directed.maximum(directed.T)
    graph.setdiag(0)
    graph.eliminate_zeros()

    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(degree)
    valid = degree > 0
    inverse_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    graph = (sp.diags(inverse_sqrt) @ graph @ sp.diags(inverse_sqrt)).tocoo()

    sparse_indices = torch.tensor(
        np.vstack([graph.row, graph.col]),
        dtype=torch.long,
        device=features.device,
    )
    sparse_values = torch.tensor(
        graph.data,
        dtype=features.dtype,
        device=features.device,
    )
    matrix = torch.sparse_coo_tensor(
        sparse_indices,
        sparse_values,
        (sample_count, sample_count),
        device=features.device,
    ).coalesce()
    diagnostics["mean_normalized_edge_weight"] = (
        float(matrix.values().mean().item()) if matrix._nnz() else 0.0
    )
    return GraphResult(
        matrix=matrix,
        backend=selected_backend,
        num_edges=int(matrix._nnz()),
        diagnostics=diagnostics,
    )
