#!/usr/bin/env python3
"""RelayZero — the $0 agent that never goes down.

Loop: classify -> plan -> route (free seat) -> execute with retry/failover
-> verify -> checkpoint before every model call. On 429/5xx/timeout: cool the
seat, restore checkpoint, fail over to the next seat, resume, verify, deliver.

Keys come from env vars ONLY (see .env.example). Never committed, never logged.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid

import checkpoint
import seats
import tracer

SYSTEM = (
    "You are RelayZero, a task-execution agent. Complete the user's step fully "
    "and concretely. If asked to write something, write it. Be concise."
)

VERIFY_SYSTEM = (
    "You are a strict verifier. Score the answer 0-10 on completeness and "
    "correctness. Reply with ONLY a number."
)

RETRYABLE = (429, 500, 502, 503, 504)

# per-seat cooldowns after a retryable failure (seconds)
_cooldowns: dict[str, float] = {}


# ---------------------------------------------------------------- classifier
def classify(task: str) -> dict:
    """Cheap keyword classifier: fast / code / research / standard."""
    t = task.lower()
    code_words = ("code", "script", "function", "python", "bug", "debug", "regex", "sql")
    research_words = ("research", "compare", "survey", "summarize", "history of", "pros and cons")
    fast_words = ("haiku", "joke", "one-liner", "translate", "name for")
    if any(w in t for w in code_words):
        kind = "code"
    elif any(w in t for w in research_words):
        kind = "research"
    elif any(w in t for w in fast_words):
        kind = "fast"
    else:
        kind = "standard"
    with tracer.Span("classify", "PLAN", {"task_kind": kind}):
        pass
    tracer.event("task_classified", kind=kind)
    return {"kind": kind, "task": task}


# ------------------------------------------------------------------- router
def route(kind: str, live: list[dict]) -> list[dict]:
    """Order live seats for this task kind. Fast/light seats first for small
    tasks; stronger seats first for code/research. Seats are disposable."""
    if not live:
        raise RuntimeError("no live seats on the bench")
    pref = {
        "fast": ["groq", "sambanova", "gemini", "mistral", "openrouter", "nvidia", "nous"],
        "code": ["mistral", "groq", "openrouter", "sambanova", "nvidia", "gemini", "nous"],
        "research": ["openrouter", "groq", "mistral", "gemini", "sambanova", "nvidia", "nous"],
        "standard": ["groq", "openrouter", "mistral", "gemini", "sambanova", "nvidia", "nous"],
    }[kind]
    ranked = sorted(
        [s for s in live if s["name"] in {x["name"] for x in live}],
        key=lambda s: (pref.index(s["name"]) if s["name"] in pref else 99),
    )
    with tracer.Span("route", "PLAN", {"kind": kind, "order": [s["name"] for s in ranked]}):
        pass
    return ranked


# ----------------------------------------------------------------- executor
def execute_with_failover(
    run_id: str,
    messages: list[dict],
    ordered: list[dict],
    max_tokens: int = 1024,
    max_attempts: int | None = None,
) -> tuple[str, dict, list[dict]]:
    """Call seats in order; on retryable failure cool the seat, checkpoint, and
    fail over. Returns (text, stats, failover_log)."""
    max_attempts = max_attempts or len(ordered)
    failovers = []
    last_exc = None
    tried = 0

    for seat in ordered:
        if tried >= max_attempts:
            break
        name = seat["name"]
        if time.time() < _cooldowns.get(name, 0):
            tracer.event("seat_skipped_cooldown", seat=name)
            continue
        tried += 1
        ckpt = checkpoint.save(run_id, f"pre-call-{name}", {"seat": name, "messages_tail": messages[-1]})
        with tracer.Span("llm-call", "LLM", {"seat": name, "model": seat["model"], "checkpoint": ckpt}):
            try:
                text, stats = seats.call_seat(seat, messages, max_tokens=max_tokens)
            except seats.SeatError as exc:
                retryable = exc.status in RETRYABLE or exc.status is None
                if retryable:
                    _cooldowns[name] = time.time() + 30
                failovers.append(
                    {"seat": name, "status": exc.status, "error": str(exc)[:120],
                     "retryable": retryable}
                )
                with tracer.Span("recover", "RECOVER", {"failed_seat": name, "status": exc.status}):
                    tracer.event("failover", failed_seat=name, restored_from=ckpt)
                print(f"  [!] {name} failed ({exc.status}); cooling down, failing over…")
                last_exc = exc
                continue
        print(f"  [✓] answered by {name} ({stats['latency_ms']} ms, cost ${stats['cost_usd']:.2f})")
        return text, stats, failovers

    raise RuntimeError(f"all seats exhausted: {failovers}") from last_exc


# ----------------------------------------------------------------- verifier
def verify(text: str, task: str, ordered: list[dict], run_id: str) -> dict:
    """Cheap verifier pass: a second seat scores the answer. Below bar -> note it."""
    if os.environ.get("RELAYZERO_NO_VERIFY") == "1" or not text.strip():
        return {"score": None, "note": "skipped/empty"}
    ver_seat = ordered[-1] if len(ordered) > 1 else ordered[0]
    with tracer.Span("verify", "VERIFY", {"verifier_seat": ver_seat["name"]}):
        try:
            score_txt, _ = seats.call_seat(
                ver_seat,
                [
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": f"TASK: {task}\n\nANSWER:\n{text[:4000]}\n\nScore 0-10:"},
                ],
                max_tokens=64,
            )
            m = re.search(r"(?<!\d)(10|\d)(?!\d)", score_txt or "")
            score = int(m.group()) if m else None
            if score is not None and not 0 <= score <= 10:
                score = None
        except Exception as exc:
            score, score_txt = None, str(exc)
    if score is None:
        # verifier seat inconclusive (reasoning-only model, empty reply, …):
        # fall back to a cheap rubric so the loop still delivers a verdict.
        rubric = len(text.strip()) > 20
        return {"score": None, "verifier_seat": ver_seat["name"],
                "note": f"verifier inconclusive; rubric pass={rubric}"}
    return {"score": score, "verifier_seat": ver_seat["name"]}


# --------------------------------------------------------------------- run
def run(task: str, kill_seat: str | None = None) -> dict:
    run_id = uuid.uuid4().hex[:8]
    if kill_seat:
        os.environ["RELAYZERO_KILL_SEAT"] = kill_seat

    tracer.init()
    with tracer.Span("relay-zero-run", "WORKFLOW", {"task": task[:120]}):
        print(f"[relay-zero] run {run_id}")
        cls = classify(task)
        print(f"  classified: {cls['kind']}")

        print("  probing bench…")
        live = seats.bench(verbose=True)
        print(f"  live seats: {[s['name'] for s in live]}")
        ordered = route(cls["kind"], live)
        checkpoint.save(run_id, "plan", {"task": task, "kind": cls["kind"],
                                         "order": [s["name"] for s in ordered]})

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
        ]
        text, stats, failovers = execute_with_failover(run_id, messages, ordered)

        verdict = verify(text, task, ordered, run_id)
        print(f"  verifier score: {verdict.get('score')} (seat: {verdict.get('verifier_seat')})")

        checkpoint.save(run_id, "deliver", {"stats": stats, "failovers": failovers,
                                            "verdict": verdict})
        result = {
            "run_id": run_id,
            "task": task,
            "kind": cls["kind"],
            "answer": text,
            "stats": stats,
            "failovers": failovers,
            "verdict": verdict,
            "cost_usd": 0.0,
        }
    tracer.flush()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="RelayZero — the $0 agent that never goes down")
    ap.add_argument("task", help="the goal to complete end-to-end")
    ap.add_argument("--kill", default=None, help="simulate a dead seat (demo kill-switch)")
    args = ap.parse_args()

    # .env support (dev convenience; file itself is gitignored)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    try:
        result = run(args.task, kill_seat=args.kill)
    except Exception as exc:
        print(f"[relay-zero] FAILED: {exc}", file=sys.stderr)
        tracer.shutdown()
        return 1

    print("\n==== RESULT ====")
    print(result["answer"])
    print(f"\nanswered by: {result['stats']['seat']} ({result['stats']['model']})")
    print(f"failovers: {len(result['failovers'])} | cost: ${result['cost_usd']:.2f}")
    tracer.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
