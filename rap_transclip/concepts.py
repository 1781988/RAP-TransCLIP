from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .data import DatasetMetadata


@dataclass(frozen=True)
class ClassConcept:
    class_token: str
    class_name: str
    context_descriptions: list[str]
    local_cues: list[str]
    semantic_group: str


def normalize_concept_key(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _fallback_concept(token: str, name: str) -> ClassConcept:
    clean = name.strip()
    return ClassConcept(
        class_token=token,
        class_name=clean,
        context_descriptions=[
            f"a remote sensing scene characterized by {clean}",
            f"the spatial layout and surrounding environment of {clean}",
        ],
        local_cues=[
            clean,
            f"local structures typical of {clean}",
            f"visual patterns associated with {clean}",
        ],
        semantic_group="mixed",
    )


def load_common_knowledge(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Concept knowledge must be a mapping: {path}")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Concept entry must be a mapping: {key}")
        normalized[normalize_concept_key(str(key))] = value
    return normalized


def build_concept_bank(
    metadata: DatasetMetadata,
    common_knowledge_path: str | Path,
    dataset_override_path: str | Path | None = None,
) -> list[ClassConcept]:
    knowledge = load_common_knowledge(common_knowledge_path)
    if dataset_override_path:
        override_path = Path(dataset_override_path)
        if override_path.exists():
            overrides = load_common_knowledge(override_path)
            knowledge.update(overrides)

    concepts: list[ClassConcept] = []
    for token, name in zip(metadata.class_tokens, metadata.class_names):
        key_candidates = [
            normalize_concept_key(name),
            normalize_concept_key(token),
        ]
        entry = next(
            (knowledge[key] for key in key_candidates if key in knowledge),
            None,
        )
        if entry is None:
            concepts.append(_fallback_concept(token, name))
            continue

        contexts = [
            str(value).strip()
            for value in entry.get("context", [])
            if str(value).strip()
        ]
        local_cues = [
            str(value).strip()
            for value in entry.get("objects", entry.get("local_cues", []))
            if str(value).strip()
        ]
        if not contexts or not local_cues:
            fallback = _fallback_concept(token, name)
            contexts = contexts or fallback.context_descriptions
            local_cues = local_cues or fallback.local_cues
        group = str(entry.get("group", "mixed")).strip().lower()
        if group not in {"object", "context", "mixed"}:
            group = "mixed"

        concepts.append(
            ClassConcept(
                class_token=token,
                class_name=name,
                context_descriptions=contexts,
                local_cues=local_cues,
                semantic_group=group,
            )
        )
    return concepts


def save_concept_bank(
    concepts: list[ClassConcept],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [asdict(item) for item in concepts],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_saved_concept_bank(
    path: str | Path,
) -> list[ClassConcept]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ClassConcept(**item) for item in payload]
