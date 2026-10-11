"""Explicit artifact paths shared by the router and its pages."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data: Path
    plastic: Path
    biomedical: Path
    chemical: Path


def settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description="One workspace for three waste auditors")
    parser.add_argument("--data", type=Path, default=ROOT/"data/synthetic")
    parser.add_argument("--plastic-run", type=Path, default=ROOT/"phantom/outputs/seed42")
    parser.add_argument("--biomed-run", type=Path, default=ROOT/"biomed/outputs/seed42")
    parser.add_argument("--chemical-run", type=Path, default=ROOT/"yieldbal/outputs/seed42")
    args, _ = parser.parse_known_args(argv)
    return Settings(args.data.resolve(), args.plastic_run.resolve(),
                    args.biomed_run.resolve(), args.chemical_run.resolve())
