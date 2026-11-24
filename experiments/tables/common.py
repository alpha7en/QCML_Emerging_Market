"""Shared loaders for the Section 4 and Supplementary S1-S3 tables. Reads only files under data/runs/."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO_ROOT / "data" / "runs"


def runs_dir() -> Path:
    """Directory holding the six benchmark suites (`data/runs/`, overridable via `RUNS=<dir>`)."""
    env = os.environ.get("RUNS")
    return Path(env).resolve() if env else DEFAULT_RUNS


def load_suite_metrics(suite: str) -> dict[str, Any]:
    path = runs_dir() / suite / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def load_suite_json(suite: str, filename: str) -> dict[str, Any]:
    path = runs_dir() / suite / filename
    return json.loads(path.read_text(encoding="utf-8"))


def load_suite_csv(suite: str, filename: str) -> list[dict[str, str]]:
    path = runs_dir() / suite / filename
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fmt_p(p: float, eps: float = 1e-4) -> str:
    return f"<{eps:g}" if p < eps else f"{p:.4f}"
