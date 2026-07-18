from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import torch

from .config import config_hash
from .feature_extraction import feature_directory, feature_variant
from .metrics import expected_calibration_error, macro_f1, top1_accuracy
from .object_context import run_object_context_inference
from .utils import l2_normalize


METHODS = [
    "global_classname",
    "global_context",
    "multicrop_classname",
    "object_only",
    "fixed_object_context",
    "object_context",
]


def _load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu").to(device)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _safe_name(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-").replace(" ", "_")


def _prepare_object_concepts(
    object_texts: torch.Tensor,
    object_mask: torch.Tensor,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Create deterministic concept-bank controls without re-encoding images."""
    mode = str(mode).lower()
    class_count = object_texts.shape[0]
    if mode == "correct":
        return object_texts, object_mask, {"concept_mode": mode}

    if mode == "shuffled":
        if class_count <= 1:
            return object_texts, object_mask, {
                "concept_mode": mode,
                "shuffle_offset": 0,
            }
        offset = int(seed) % (class_count - 1) + 1
        permutation = torch.arange(
            class_count,
            device=object_texts.device,
        ).roll(offset)
        return (
            object_texts[permutation],
            object_mask[permutation],
            {
                "concept_mode": mode,
                "shuffle_offset": offset,
            },
        )

    if mode == "generic":
        valid = object_texts[object_mask.bool()]
        if valid.numel() == 0:
            raise ValueError("Cannot build generic concepts from an empty bank")
        generic = l2_normalize(valid.float().mean(dim=0, keepdim=True)).to(
            dtype=object_texts.dtype
        )
        generic = generic.view(1, 1, -1).expand(
            class_count,
            1,
            -1,
        ).contiguous()
        generic_mask = torch.ones(
            class_count,
            1,
            dtype=torch.bool,
            device=object_mask.device,
        )
        return generic, generic_mask, {"concept_mode": mode}

    raise ValueError(
        f"Unsupported inference.object_concept_mode={mode!r}; "
        "expected correct, shuffled, or generic"
    )


def run_experiment(
    cfg: dict,
    dataset_name: str,
    model_name: str,
    architecture: str,
    method: str,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    device_name = cfg["project"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    feature_dir = feature_directory(
        cfg,
        dataset_name,
        model_name,
        architecture,
    )
    global_features = _load_tensor(
        feature_dir / "global_images.pt",
        device,
    ).float()
    local_features = _load_tensor(
        feature_dir / "local_images.pt",
        device,
    ).float()
    labels = torch.load(
        feature_dir / "labels.pt",
        map_location="cpu",
    ).long().to(device)
    class_texts = _load_tensor(
        feature_dir / "class_texts.pt",
        device,
    ).float()
    context_texts = _load_tensor(
        feature_dir / "context_texts.pt",
        device,
    ).float()
    object_texts = _load_tensor(
        feature_dir / "object_texts.pt",
        device,
    ).float()
    object_mask = _load_tensor(
        feature_dir / "object_mask.pt",
        device,
    ).bool()

    inference_cfg = cfg.get("inference", {})
    concept_mode = str(inference_cfg.get("object_concept_mode", "correct"))
    object_texts, object_mask, concept_diagnostics = _prepare_object_concepts(
        object_texts,
        object_mask,
        concept_mode,
        int(inference_cfg.get("concept_shuffle_seed", 17)),
    )

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    output = run_object_context_inference(
        method,
        global_features,
        local_features,
        class_texts,
        context_texts,
        object_texts,
        object_mask,
        cfg,
    )
    elapsed = time.perf_counter() - started
    probabilities = output.probabilities

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if torch.cuda.is_available() and device.type == "cuda"
        else 0.0
    )

    runtime_cfg = cfg.get("runtime", {})
    experiment_tag = str(
        runtime_cfg.get("experiment_tag", "object_context")
    )
    diagnostics = {**output.diagnostics, **concept_diagnostics}
    row = {
        "dataset": dataset_name,
        "model": model_name,
        "architecture": architecture,
        "feature_variant": feature_variant(cfg),
        "method": method,
        "experiment_tag": experiment_tag,
        "object_concept_mode": concept_mode,
        "object_view_topk": int(inference_cfg.get("object_view_topk", 2)),
        "class_consensus_view_topk": int(
            inference_cfg.get("class_consensus_view_topk", 2)
        ),
        "num_samples": int(len(labels)),
        "num_classes": int(class_texts.shape[0]),
        "num_local_views": int(local_features.shape[1]),
        "top1": round(top1_accuracy(probabilities, labels), 4),
        "macro_f1": round(macro_f1(probabilities, labels), 4),
        "ece": round(expected_calibration_error(probabilities, labels), 4),
        "inference_seconds": round(float(elapsed), 4),
        "peak_cuda_memory_mb": round(float(peak_memory_mb), 2),
        "config_hash": config_hash(cfg),
        "diagnostics": json.dumps(
            diagnostics,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    result_root = Path(cfg["paths"]["results"])
    _append_csv(result_root / "raw_results.csv", row)

    if runtime_cfg.get("save_predictions", False):
        prediction_dir = result_root / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        stem = "__".join(
            [
                _safe_name(dataset_name),
                _safe_name(model_name),
                _safe_name(architecture),
                _safe_name(feature_variant(cfg)),
                _safe_name(method),
                _safe_name(experiment_tag),
            ]
        )
        torch.save(
            {
                "probabilities": probabilities.detach().cpu(),
                "scores": output.scores.detach().cpu(),
                "object_weights": (
                    None
                    if output.object_weights is None
                    else output.object_weights.detach().cpu()
                ),
                "artifacts": {
                    key: value.detach().cpu()
                    for key, value in output.artifacts.items()
                },
                "labels": labels.detach().cpu(),
                "diagnostics": diagnostics,
                "experiment_tag": experiment_tag,
                "object_concept_mode": concept_mode,
            },
            prediction_dir / f"{stem}.pt",
        )

    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row
