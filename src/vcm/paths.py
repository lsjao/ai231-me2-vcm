"""Project-root-anchored default paths, so commands work from any directory
(run.cmd runs from src/, the Pi setup runs from the repo root, tests from src/)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def root_path(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))
