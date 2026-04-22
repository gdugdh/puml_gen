from __future__ import annotations

import argparse
from pathlib import Path

from src.generator import generate_from_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="puml_gen",
        description="Generate PlantUML activity diagrams from synthetic IR using LangGraph and OpenRouter.",
    )
    parser.add_argument("--input", required=True, help="Path to synthetic IR JSON.")
    parser.add_argument(
        "--outdir",
        default="output",
        help="Directory for generated .puml files.",
    )
    parser.add_argument(
        "--diagram-mode",
        choices=["route", "service"],
        default="route",
        help="Diagram detail mode.",
    )
    args = parser.parse_args(argv)

    generated_files = generate_from_file(
        Path(args.input),
        Path(args.outdir),
        diagram_mode=args.diagram_mode,
    )
    for path in generated_files:
        print(path.resolve().as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
