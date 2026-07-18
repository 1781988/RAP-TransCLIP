#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rap_transclip.concepts import build_concept_bank, save_concept_bank
from rap_transclip.config import load_config
from rap_transclip.data import load_dataset_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata = load_dataset_metadata(
        cfg["paths"]["datasets"],
        args.dataset,
    )
    concept_cfg = cfg["concept_bank"]
    pattern = concept_cfg.get("dataset_override_pattern")
    override = (
        str(pattern).format(dataset=args.dataset)
        if pattern
        else None
    )
    concepts = build_concept_bank(
        metadata,
        concept_cfg["common_knowledge"],
        override,
    )
    output = Path(
        args.output
        or (
            Path(cfg["paths"]["results"])
            / "concept_banks"
            / f"{args.dataset}.json"
        )
    )
    save_concept_bank(concepts, output)
    print(output)


if __name__ == "__main__":
    main()
