from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import torch

from .config import config_hash
from .metrics import expected_calibration_error, macro_f1, top1_accuracy
from .protocols import create_protocol
from .solver import (
    solve_rap_transclip,
    solve_rs_transclip,
    solve_shift_aware_rap_transclip,
    zero_shot_assignments,
)
from .utils import l2_normalize


def _load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu").float().to(device)


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


def _save_solver_bundle(
    cfg: dict,
    dataset_name: str,
    model_name: str,
    architecture: str,
    method: str,
    protocol_name: str,
    seed: int,
    output,
) -> None:
    result_root = Path(cfg["paths"]["results"])
    bundle_dir = result_root / "diagnostics"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tag = str(cfg.get("runtime", {}).get("experiment_tag", "main"))
    name = "__".join(
        _safe_name(item)
        for item in (
            dataset_name,
            model_name,
            architecture,
            method,
            protocol_name,
            f"seed{seed}",
            tag,
        )
    )
    torch.save(
        {
            "assignments": output.assignments.detach().cpu(),
            "prototypes": output.prototypes.detach().cpu(),
            "prompt_weights": (
                None
                if output.prompt_weights is None
                else output.prompt_weights.detach().cpu()
            ),
            "class_prior": output.class_prior.detach().cpu(),
            "sample_reliability": (
                None
                if output.sample_reliability is None
                else output.sample_reliability.detach().cpu()
            ),
            "iterations": int(output.iterations),
            "elapsed_seconds": float(output.elapsed_seconds),
            "diagnostics": output.diagnostics,
        },
        bundle_dir / f"{name}.pt",
    )


def run_experiment(
    cfg: dict,
    dataset_name: str,
    model_name: str,
    architecture: str,
    method: str,
    protocol_name: str = "full",
    protocol_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device_name = cfg["project"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    feature_dir = (
        Path(cfg["paths"]["features"])
        / dataset_name
        / model_name
        / architecture
    )

    images = _load_tensor(feature_dir / "images.pt", device)
    labels = torch.load(
        feature_dir / "labels.pt",
        map_location="cpu",
    ).long()
    texts_all = _load_tensor(feature_dir / "texts_all.pt", device)
    texts_uniform = _load_tensor(
        feature_dir / "texts_uniform.pt",
        device,
    )

    seed = int(
        cfg["protocol"].get(
            "seed",
            cfg["project"].get("seed", 1),
        )
    )
    protocol = create_protocol(
        protocol_name,
        labels,
        seed,
        **(protocol_args or {}),
    )
    idx_cpu = protocol.indices.long()
    images = images[idx_cpu.to(device)]
    labels = labels[idx_cpu].to(device)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    output = None
    if method == "zero_shot":
        probabilities = zero_shot_assignments(
            l2_normalize(images),
            l2_normalize(texts_uniform),
            float(cfg["zero_shot"]["logit_scale"]),
        )
        solver_time = time.perf_counter() - started
        iterations = 0
        diagnostics = {}
    elif method == "rs_transclip":
        output = solve_rs_transclip(images, texts_uniform, cfg)
        probabilities = output.assignments
        solver_time = output.elapsed_seconds
        iterations = output.iterations
        diagnostics = output.diagnostics
    elif method == "rap_transclip":
        output = solve_rap_transclip(images, texts_all, cfg)
        probabilities = output.assignments
        solver_time = output.elapsed_seconds
        iterations = output.iterations
        diagnostics = output.diagnostics
    elif method in {
        "sa_rap_transclip",
        "shift_aware_rap_transclip",
    }:
        output = solve_shift_aware_rap_transclip(
            images,
            texts_all,
            texts_uniform,
            cfg,
        )
        probabilities = output.assignments
        solver_time = output.elapsed_seconds
        iterations = output.iterations
        diagnostics = output.diagnostics
    else:
        raise ValueError(f"Unsupported method: {method}")

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if torch.cuda.is_available() and device.type == "cuda"
        else 0.0
    )

    runtime_cfg = cfg.get("runtime", {})
    if output is not None and runtime_cfg.get("save_diagnostics", False):
        _save_solver_bundle(
            cfg,
            dataset_name,
            model_name,
            architecture,
            method,
            protocol_name,
            seed,
            output,
        )
    elif (
        output is not None
        and runtime_cfg.get("save_assignments", False)
    ):
        assignment_dir = Path(cfg["paths"]["results"]) / "assignments"
        assignment_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            probabilities.detach().cpu(),
            assignment_dir
            / (
                f"{_safe_name(dataset_name)}_"
                f"{_safe_name(model_name)}_"
                f"{_safe_name(architecture)}_"
                f"{_safe_name(method)}_"
                f"{_safe_name(protocol_name)}_seed{seed}.pt"
            ),
        )

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "architecture": architecture,
        "method": method,
        "protocol": protocol_name,
        "experiment_tag": str(runtime_cfg.get("experiment_tag", "main")),
        "seed": int(cfg["project"].get("seed", seed)),
        "num_samples": int(len(labels)),
        "num_candidate_classes": int(texts_all.shape[1]),
        "top1": round(top1_accuracy(probabilities, labels), 4),
        "macro_f1": round(macro_f1(probabilities, labels), 4),
        "ece": round(expected_calibration_error(probabilities, labels), 4),
        "solver_seconds": round(float(solver_time), 4),
        "peak_cuda_memory_mb": round(float(peak_memory_mb), 2),
        "iterations": int(iterations),
        "config_hash": config_hash(cfg),
        "protocol_metadata": json.dumps(
            protocol.metadata,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "diagnostics": json.dumps(
            diagnostics,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    _append_csv(Path(cfg["paths"]["results"]) / "raw_results.csv", row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row
