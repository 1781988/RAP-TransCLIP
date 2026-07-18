from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .concepts import build_concept_bank, save_concept_bank
from .data import MultiViewImageDataset, load_dataset_metadata
from .models import load_vlm
from .multiview import build_view_specs
from .utils import l2_normalize, write_json


FEATURE_VERSION = "object-context-v1"


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def feature_variant(cfg: dict) -> str:
    value = str(cfg["feature_extraction"].get("variant", "clean")).strip()
    return value or "clean"


def feature_directory(
    cfg: dict,
    dataset_name: str,
    model_name: str,
    architecture: str,
) -> Path:
    return (
        Path(cfg["paths"]["features"])
        / dataset_name
        / model_name
        / architecture
        / feature_variant(cfg)
    )


def _encode_text_batches(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start : start + batch_size]).to(device)
            with _autocast_context(device, amp):
                features = model.encode_text(tokens)
            outputs.append(l2_normalize(features.float()).cpu())
    return torch.cat(outputs, dim=0)


def _average_prompted_concepts(
    model,
    tokenizer,
    concepts: Iterable[str],
    templates: list[str],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    concepts = list(concepts)
    prompted = [
        template.format(text=concept)
        for concept in concepts
        for template in templates
    ]
    encoded = _encode_text_batches(
        model,
        tokenizer,
        prompted,
        device,
        batch_size,
        amp,
    )
    encoded = encoded.view(len(concepts), len(templates), -1).mean(dim=1)
    return l2_normalize(encoded)


def _extract_text_features(
    cfg: dict,
    loaded,
    concepts,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    semantic_cfg = cfg["semantic_prompts"]
    batch_size = int(semantic_cfg.get("batch_size", 256))
    amp = bool(cfg["feature_extraction"].get("amp", True))

    class_texts = _average_prompted_concepts(
        loaded.model,
        loaded.tokenizer,
        [item.class_name for item in concepts],
        list(semantic_cfg["class_templates"]),
        device,
        batch_size,
        amp,
    )
    context_texts = []
    object_features: list[torch.Tensor] = []
    max_objects = max(len(item.local_cues) for item in concepts)

    for item in tqdm(concepts, desc="semantic text concepts"):
        context_prompt_features = _average_prompted_concepts(
            loaded.model,
            loaded.tokenizer,
            item.context_descriptions,
            list(semantic_cfg["context_templates"]),
            device,
            batch_size,
            amp,
        )
        context_texts.append(
            l2_normalize(context_prompt_features.mean(dim=0, keepdim=True))[0]
        )

        cue_features = _average_prompted_concepts(
            loaded.model,
            loaded.tokenizer,
            item.local_cues,
            list(semantic_cfg["object_templates"]),
            device,
            batch_size,
            amp,
        )
        padded = torch.zeros(
            max_objects,
            cue_features.shape[1],
            dtype=cue_features.dtype,
        )
        padded[: len(item.local_cues)] = cue_features
        object_features.append(padded)

    object_mask = torch.zeros(
        len(concepts),
        max_objects,
        dtype=torch.bool,
    )
    for class_id, item in enumerate(concepts):
        object_mask[class_id, : len(item.local_cues)] = True

    return (
        class_texts,
        l2_normalize(torch.stack(context_texts)),
        torch.stack(object_features),
        object_mask,
    )


def extract_features(
    cfg: dict,
    dataset_name: str,
    model_name: str,
    architecture: str,
    overwrite: bool = False,
) -> Path:
    device_name = cfg["project"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model_cfg = cfg["models"][model_name]
    arch_cfg = model_cfg["architectures"][architecture]
    loaded = load_vlm(
        model_cfg["backend"],
        architecture,
        arch_cfg.get("pretrained"),
        arch_cfg.get("checkpoint"),
        device,
    )

    output_dir = feature_directory(
        cfg,
        dataset_name,
        model_name,
        architecture,
    )
    marker = output_dir / "metadata.json"
    required = [
        output_dir / "global_images.pt",
        output_dir / "local_images.pt",
        output_dir / "class_texts.pt",
        output_dir / "context_texts.pt",
        output_dir / "object_texts.pt",
        output_dir / "object_mask.pt",
    ]
    if (
        marker.exists()
        and all(path.exists() for path in required)
        and not overwrite
    ):
        print(f"Object-context features already exist; skipping: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_dataset_metadata(
        cfg["paths"]["datasets"],
        dataset_name,
    )
    concept_cfg = cfg["concept_bank"]
    override_pattern = concept_cfg.get("dataset_override_pattern")
    override_path = (
        str(override_pattern).format(dataset=dataset_name)
        if override_pattern
        else None
    )
    concepts = build_concept_bank(
        metadata,
        concept_cfg["common_knowledge"],
        override_path,
    )
    save_concept_bank(concepts, output_dir / "concept_bank.json")

    view_cfg = cfg["local_views"]
    view_specs = build_view_specs(
        view_cfg["scales"],
        view_cfg["positions"],
    )
    index_path = (
        Path(cfg["paths"]["indexes"])
        / f"{dataset_name}.jsonl"
    )
    dataset = MultiViewImageDataset(
        index_path,
        loaded.preprocess,
        view_specs,
        downsample_factor=int(
            cfg["feature_extraction"].get("downsample_factor", 1)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["feature_extraction"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["feature_extraction"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    global_features: list[torch.Tensor] = []
    local_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    image_paths: list[str] = []
    started = time.perf_counter()
    amp = bool(cfg["feature_extraction"].get("amp", True))

    with torch.inference_mode():
        for (
            global_images,
            local_images,
            batch_labels,
            batch_paths,
        ) in tqdm(loader, desc="multi-view image features"):
            batch_size, view_count = local_images.shape[:2]
            global_images = global_images.to(device, non_blocking=True)
            local_images = local_images.flatten(0, 1).to(
                device,
                non_blocking=True,
            )
            with _autocast_context(device, amp):
                global_batch = loaded.model.encode_image(global_images)
                local_batch = loaded.model.encode_image(local_images)
            global_batch = l2_normalize(global_batch.float()).cpu()
            local_batch = l2_normalize(local_batch.float()).view(
                batch_size,
                view_count,
                -1,
            ).cpu()
            global_features.append(global_batch)
            local_features.append(local_batch)
            labels.append(batch_labels.long().cpu())
            image_paths.extend(batch_paths)

    all_global = torch.cat(global_features)
    all_local = torch.cat(local_features)
    all_labels = torch.cat(labels)

    (
        class_texts,
        context_texts,
        object_texts,
        object_mask,
    ) = _extract_text_features(
        cfg,
        loaded,
        concepts,
        device,
    )

    storage_dtype = str(
        cfg["feature_extraction"].get("storage_dtype", "float16")
    )
    if storage_dtype == "float16":
        all_global = all_global.half()
        all_local = all_local.half()
        class_texts = class_texts.half()
        context_texts = context_texts.half()
        object_texts = object_texts.half()
    elif storage_dtype != "float32":
        raise ValueError(
            "feature_extraction.storage_dtype must be float16 or float32"
        )

    torch.save(all_global, output_dir / "global_images.pt")
    torch.save(all_local, output_dir / "local_images.pt")
    torch.save(all_labels, output_dir / "labels.pt")
    torch.save(class_texts, output_dir / "class_texts.pt")
    torch.save(context_texts, output_dir / "context_texts.pt")
    torch.save(object_texts, output_dir / "object_texts.pt")
    torch.save(object_mask, output_dir / "object_mask.pt")
    write_json(
        output_dir / "classes.json",
        {
            "tokens": metadata.class_tokens,
            "names": metadata.class_names,
            "semantic_groups": [
                item.semantic_group for item in concepts
            ],
        },
    )
    write_json(output_dir / "image_paths.json", image_paths)
    write_json(
        marker,
        {
            "feature_version": FEATURE_VERSION,
            "variant": feature_variant(cfg),
            "dataset": dataset_name,
            "model": model_name,
            "architecture": architecture,
            "num_images": int(all_global.shape[0]),
            "num_classes": int(class_texts.shape[0]),
            "num_local_views": int(all_local.shape[1]),
            "local_view_names": [spec.name for spec in view_specs],
            "embedding_dim": int(all_global.shape[1]),
            "downsample_factor": int(
                cfg["feature_extraction"].get("downsample_factor", 1)
            ),
            "storage_dtype": storage_dtype,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    return output_dir
