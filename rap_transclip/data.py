from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .multiview import (
    ViewSpec,
    apply_resolution_degradation,
    deterministic_local_crops,
)


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_name: str
    class_tokens: list[str]
    class_names: list[str]


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def load_dataset_metadata(
    dataset_dir: str | Path,
    dataset_name: str,
) -> DatasetMetadata:
    root = Path(dataset_dir) / dataset_name
    tokens = _read_nonempty_lines(root / "classes.txt")
    names = _read_nonempty_lines(root / "class_changes.txt")
    if len(tokens) != len(names):
        raise ValueError(
            f"{dataset_name}: classes.txt has {len(tokens)} entries but "
            f"class_changes.txt has {len(names)} entries"
        )
    return DatasetMetadata(dataset_name, tokens, names)


def _infer_class_token(
    path: Path,
    image_root: Path,
    class_tokens: set[str],
) -> str:
    relative = path.relative_to(image_root)
    if len(relative.parts) > 1 and relative.parts[0] in class_tokens:
        return relative.parts[0]
    stem = path.stem.lower()
    candidates = [
        token
        for token in class_tokens
        if stem == token or stem.startswith(token + "_")
    ]
    if candidates:
        return max(candidates, key=len)
    prefix = stem.rsplit("_", 1)[0]
    if prefix in class_tokens:
        return prefix
    raise ValueError(f"Cannot map image to class token: {path}")


def build_index(
    datasets_root: str | Path,
    dataset_name: str,
    output_path: str | Path,
    extensions: Iterable[str],
) -> int:
    metadata = load_dataset_metadata(datasets_root, dataset_name)
    root = Path(datasets_root) / dataset_name
    image_root = root / "images"
    if not image_root.exists():
        raise FileNotFoundError(image_root)

    normalized_ext = {ext.lower() for ext in extensions}
    token_to_label = {
        token: idx for idx, token in enumerate(metadata.class_tokens)
    }
    records: list[dict] = []
    failures: list[str] = []

    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_ext:
            continue
        try:
            token = _infer_class_token(
                path,
                image_root,
                set(metadata.class_tokens),
            )
            with Image.open(path) as image:
                image.verify()
            label = token_to_label[token]
            records.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "class_token": token,
                    "class_name": metadata.class_names[label],
                }
            )
        except Exception as exc:
            failures.append(f"{path}: {exc}")

    if failures:
        raise RuntimeError(
            f"Indexing failed for {len(failures)} files. First failures:\n"
            + "\n".join(failures[:20])
        )
    if not records:
        raise RuntimeError(f"No supported images found in {image_root}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


class MultiViewImageDataset(Dataset):
    """Return one global image tensor and deterministic local crop tensors."""

    def __init__(
        self,
        index_path: str | Path,
        transform,
        view_specs: list[ViewSpec],
        downsample_factor: int = 1,
    ):
        self.transform = transform
        self.view_specs = list(view_specs)
        self.downsample_factor = max(1, int(downsample_factor))
        self.records = [
            json.loads(line)
            for line in Path(index_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"Empty index: {index_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image = apply_resolution_degradation(
                image,
                self.downsample_factor,
            )
            global_tensor = self.transform(image)
            local_tensors = torch.stack(
                [
                    self.transform(crop)
                    for crop in deterministic_local_crops(
                        image,
                        self.view_specs,
                    )
                ]
            )
        return (
            global_tensor,
            local_tensors,
            int(record["label"]),
            str(record["path"]),
        )
