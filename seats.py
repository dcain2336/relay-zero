"""Free model bench. Seats are disposable: bench() probes every seat at startup
and the router only touches seats that answer. Keys come from env vars ONLY."""

from __future__ import annotations

import os
import time

import requests

TIMEOUT = 25
PROBE_PROMPT = "Reply with exactly: OK"

SEATS = [
    {
        "name": "groq",
        "provider": "Groq",
        "key_env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "headers": lambda k: {"Authorization": f"Bearer {k}"},
    },
    {
        "name": "mistral",
        "provider": "Mistral AI",
        "key_env": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "headers": lambda k: {"Authorization": f"Bearer {k}"},
    },
    {
        "name": "nous",
        "provider": "Nous Research",
        "key_env": "NOUS_API_KEY",
        "url": "https://api.nousresearch.com/v1/chat/completions",
        "model": "DeepHermes-3-Llama-3-8B-Preview",
        "headers": lambda k: {"Authorization": f"Bearer {k}"},
    },
    {
        "name": "nvidia",
        "provider": "NVIDIA NIM",
        "key_env": "NVIDIA_API_KEY",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "meta/llama-3.3-70b-instruct",
        "headers": lambda k: {"Authorization": f"Bearer {k}"},
    },
    {
        "name": "openrouter",
        "provider": "OpenRouter",
        "key_env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "headers": lambda k: {
            "Authorization": f"Bearer {k}",
            "HTTP-Referer": "https://github.com/dcain2336/relay-zero",
            "X-Title": "RelayZero-neatHack",
        },
    },
    {
        "name": "sambanova",
        "provider": "SambaNova",
        "key_env": "SAMBANOVA_API_KEY",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "model": "Meta-Llama-3.3-70B-Instruct",
        "headers": lambda k: {"Authorization": f"Bearer {k}"},
    },
]

# Gemini uses its own REST shape (v1beta generateContent, key as query param).
GEMINI_SEAT = {
    "name": "gemini",
    "provider": "Google AI",
    "key_env": "GEMINI_API_KEY",
    "model": "gemini-2.0-flash",
}


def _killed(name: str) -> bool:
    """Demo kill-switch: RELAYZERO_KILL_SEAT=<seat> simulates a dead seat."""
    return os.environ.get("RELAYZERO_KILL_SEAT", "").strip().lower() == name.lower()


class SeatError(Exception):
    def __init__(self, seat: str, status: int | None, message: str):
        super().__init__(f"[{seat}] {status or 'conn'}: {message}")
        self.seat = seat
        self.status = status


def call_seat(seat: dict, messages: list[dict], max_tokens: int = 1024) -> tuple[str, dict]:
    """Call one seat. Returns (text, stats). Raises SeatError on any failure."""
    name = seat["name"]
    if _killed(name):
        raise SeatError(name, 500, "simulated kill (RELAYZERO_KILL_SEAT)")

    key = os.environ.get(seat["key_env"], "").strip()
    if not key:
        raise SeatError(name, None, f"no key in {seat['key_env']}")

    if name == "gemini":
        return _call_gemini(seat, key, messages, max_tokens)

    start = time.perf_counter()
    try:
        resp = requests.post(
            seat["url"],
            headers={**seat["headers"](key), "Content-Type": "application/json"},
            json={
                "model": seat["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SeatError(name, None, f"connection failed: {exc}")
    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        raise SeatError(name, resp.status_code, resp.text[:300])
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise SeatError(name, resp.status_code, f"unexpected payload: {str(data)[:200]}")
    usage = data.get("usage", {})
    stats = {
        "seat": name,
        "model": seat["model"],
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": 0.0,
    }
    return text, stats


def _call_gemini(seat: dict, key: str, messages: list[dict], max_tokens: int):
    name = seat["name"]
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{seat['model']}:generateContent?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "role": "user" if m["role"] == "user" else "model",
                        "parts": [{"text": m["content"]}],
                    }
                    for m in messages
                    if m["role"] in ("user", "assistant")
                ],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SeatError(name, None, f"connection failed: {exc}")
    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        raise SeatError(name, resp.status_code, resp.text[:300])
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise SeatError(name, resp.status_code, f"unexpected payload: {str(data)[:200]}")
    stats = {
        "seat": name,
        "model": seat["model"],
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": 0.0,
    }
    return text, stats


def all_seats() -> list[dict]:
    return SEATS + [GEMINI_SEAT]


def bench(verbose: bool = False) -> list[dict]:
    """Probe every seat; return the live ones in probe order."""
    live = []
    for seat in all_seats():
        name = seat["name"]
        if _killed(name):
            if verbose:
                print(f"  [bench] {name}: KILLED (simulated)")
            continue
        try:
            text, stats = call_seat(
                seat, [{"role": "user", "content": PROBE_PROMPT}], max_tokens=8
            )
            if verbose:
                print(f"  [bench] {name}: LIVE ({stats['latency_ms']} ms, said {text.strip()[:20]!r})")
            live.append(seat)
        except SeatError as exc:
            if verbose:
                print(f"  [bench] {name}: dead — {exc}")
    return live
