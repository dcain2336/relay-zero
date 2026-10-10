"""Entire-style checkpoints: persist agent state before every model call so a
failover restores instead of restarting. JSON snapshots under .checkpoints/<run_id>/.
Also exposes entire_status() to verify the Entire CLI hooks are capturing."""
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".checkpoints")


def _run_dir(run_id: str) -> str:
    d = os.path.join(_BASE, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def save(run_id: str, step_name: str, state: dict) -> str:
    """Write a checkpoint snapshot. Returns the checkpoint path."""
    d = _run_dir(run_id)
    payload = {
        "run_id": run_id,
        "step": step_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
    }
    idx = len([f for f in os.listdir(d) if f.endswith(".json")])
    path = os.path.join(d, f"{idx:03d}-{step_name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def latest(run_id: str) -> dict | None:
    """Load the most recent checkpoint for a run, or None."""
    d = os.path.join(_BASE, run_id)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    if not files:
        return None
    with open(os.path.join(d, files[-1])) as fh:
        return json.load(fh)


def purge(run_id: str):
    shutil.rmtree(os.path.join(_BASE, run_id), ignore_errors=True)


def entire_status(repo_dir: str) -> str:
    """Ask the Entire CLI for its view of this repo (empty string if absent)."""
    try:
        out = subprocess.run(
            ["entire", "status"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (out.stdout or out.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
