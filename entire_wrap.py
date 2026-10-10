"""Entire session wrapper: wraps a RelayZero run in Entire's agent-session
lifecycle so every agent commit becomes a replayable checkpoint.

Entire's git hooks only attach a checkpoint (Entire-Checkpoint: trailer +
refs/entire/checkpoints/* ref) when an agent session is ACTIVE at commit time.
This module fires the headless session lifecycle (no real Claude Code needed):

    session_start(run_id, task) -> ... agent works, commits ... -> session_stop(run_id)

Transcript turns are written as JSONL to the given transcript path; Entire's
`checkpoint explain` renders them as the session's reasoning evidence.
All commands are local and need NO Entire login.
"""

from __future__ import annotations

import json
import os
import subprocess

ENTIRE = os.path.expanduser("~/.local/bin/entire")


def _hook(sub: str, payload: dict) -> bool:
    """Fire `entire hooks claude-code <sub>` with a JSON payload on stdin."""
    try:
        r = subprocess.run(
            [ENTIRE, "hooks", "claude-code", sub],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def session_start(run_id: str, task: str, transcript_path: str, cwd: str) -> bool:
    ok = _hook("session-start", {
        "session_id": f"relayzero-{run_id}",
        "transcript_path": transcript_path,
        "cwd": cwd,
    })
    ok &= _hook("user-prompt-submit", {
        "session_id": f"relayzero-{run_id}",
        "transcript_path": transcript_path,
        "cwd": cwd,
        "prompt": task,
    })
    return ok


def session_stop(run_id: str, transcript_path: str, cwd: str) -> bool:
    return _hook("stop", {
        "session_id": f"relayzero-{run_id}",
        "transcript_path": transcript_path,
        "cwd": cwd,
    })


def append_turn(transcript_path: str, role: str, text: str):
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "a") as fh:
        fh.write(json.dumps({"role": role, "content": text[:2000]}) + "\n")


def agent_commit(repo_dir: str, message: str) -> tuple[bool, str]:
    """Commit the agent's artifacts; Entire hooks attach the checkpoint."""
    try:
        subprocess.run(["git", "-C", repo_dir, "add", "-A"], check=True, timeout=30)
        r = subprocess.run(["git", "-C", repo_dir, "commit", "-m", message],
                           capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0
        out = (r.stdout + r.stderr).strip()
        if "nothing to commit" in out:
            return True, "nothing to commit"
        return ok, out[-300:]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)[:200]


def checkpoint_list(repo_dir: str) -> str:
    try:
        r = subprocess.run([ENTIRE, "checkpoint", "list"], cwd=repo_dir,
                           capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def checkpoint_explain(repo_dir: str, checkpoint_id: str) -> str:
    try:
        r = subprocess.run([ENTIRE, "checkpoint", "explain", checkpoint_id],
                           cwd=repo_dir, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
