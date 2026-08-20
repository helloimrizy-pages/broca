"""Shared utilities: seeding, device resolution, provenance, results IO.

Every experiment in this repo writes a JSON file under ``results/`` that carries
enough provenance to reproduce it: the full config, the git commit, the seed,
wall clock time, and the pinned package versions.  Nothing is ever hand-edited
into those files.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(name: str) -> torch.device:
    """Resolve a ``--device`` flag value to a torch device, failing loudly.

    We never fall back silently: if the user asks for ``cuda`` or ``mps`` and it
    is unavailable, that is a configuration error, not something to paper over.
    """
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but torch.backends.mps.is_available() is False")
    if name not in ("cuda", "mps", "cpu"):
        raise ValueError(f"unsupported device {name!r}; use cuda, mps, cpu or auto")
    return torch.device(name)


def git_commit() -> dict[str, Any]:
    """Return the current commit hash and whether the tree is dirty."""
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def package_versions() -> dict[str, str]:
    """Versions of every package whose behaviour can move a number."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ("torch", "transformers", "datasets", "scipy", "numpy", "safetensors",
                "tokenizers", "accelerate", "lm_eval"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    return versions


def device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    elif device.type == "mps":
        info["name"] = platform.processor() or "apple-silicon"
        try:
            info["recommended_max_working_set_bytes"] = torch.mps.recommended_max_memory()
        except Exception:
            pass
    return info


@dataclass
class RunRecord:
    """Container for one experiment's provenance plus its raw metrics."""

    name: str
    config: dict[str, Any]
    seed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    wall_clock_seconds: float | None = None
    started_at: str | None = None
    git: dict[str, Any] = field(default_factory=git_commit)
    versions: dict[str, str] = field(default_factory=package_versions)
    device: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = RESULTS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=_json_default))
        return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


class Timer:
    """Wall clock timer used for the ``wall_clock_seconds`` field."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.time() - self.start
