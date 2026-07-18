from __future__ import annotations

import argparse
from pathlib import Path

from .config import apply_overrides, load_config
from .data import build_index
from .feature_extraction import extract_features
from .runner import METHODS, run_experiment


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/standard.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Nested key=value override",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="object-context-clip")
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Build one dataset index")
    _common_config(index_parser)
    index_parser.add_argument("--dataset", required=True)

    extract_parser = sub.add_parser(
        "extract",
        help="Extract global, local, and semantic text features",
    )
    _common_config(extract_parser)
    extract_parser.add_argument("--dataset", required=True)
    extract_parser.add_argument("--model", required=True)
    extract_parser.add_argument("--architecture", required=True)
    extract_parser.add_argument("--overwrite", action="store_true")

    run_parser = sub.add_parser("run", help="Run one inference method")
    _common_config(run_parser)
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--architecture", required=True)
    run_parser.add_argument("--method", choices=METHODS, required=True)

    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)

    if args.command == "index":
        output = Path(cfg["paths"]["indexes"]) / f"{args.dataset}.jsonl"
        count = build_index(
            cfg["paths"]["datasets"],
            args.dataset,
            output,
            cfg["feature_extraction"]["image_extensions"],
        )
        print(f"Indexed {count} images -> {output}")
    elif args.command == "extract":
        output = extract_features(
            cfg,
            args.dataset,
            args.model,
            args.architecture,
            args.overwrite,
        )
        print(output)
    elif args.command == "run":
        run_experiment(
            cfg,
            args.dataset,
            args.model,
            args.architecture,
            args.method,
        )


if __name__ == "__main__":
    main()
