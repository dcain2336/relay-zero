# RelayZero — the $0 agent that never goes down

**Team ArcOmega · neatHack 2026 entry**

> Every model is free. Every model will fail. The agent finishes anyway.

RelayZero is a single end-to-end agent that completes tasks on **$0 of inference**
by routing across a bench of **free model seats** with live failover. Give it a
goal, kill its model mid-run — it cools the seat down, restores state from an
[Entire](https://entire.io) checkpoint, fails over to the next free seat, verifies
its own answer, and finishes.

## The loop

```
plan → route → execute → verify
            ↑                |
            └─ failover ←----+  (checkpoint → cool seat → restore → retry)
```

Every run is traced end-to-end in [neatlogs](https://neatlogs.com) (spans for
plan / route / llm-call / verify / recover, with latency, tokens, cost=$0.00).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your FREE seat keys (see below)
python3 relay.py "Research three free LLM observability tools and compare them"
```

Kill-test (the demo centerpiece):

```bash
RELAYZERO_KILL_SEAT=groq python3 relay.py "Write a haiku about failing over, then explain the poem"
```

## Free seats

| Seat | Provider | Endpoint | Free model (verified 2026-10-10) |
|------|----------|----------|----------------------------------|
| groq | Groq | api.groq.com/openai/v1 | qwen/qwen3.8-27b |
| openrouter | OpenRouter | openrouter.ai/api/v1 | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free |
| gemini | Google AI | generativelanguage.googleapis.com | gemini-flash-latest |
| mistral | Mistral AI | api.mistral.ai/v1 | mistral-small-latest (rate-limited at bench time) |
| nous | Nous Research | inference-api.nousresearch.com | needs account-level user tag (bench skips) |
| nvidia | NVIDIA NIM | integrate.api.nvidia.com/v1 | key not provisioned for chat models (bench skips) |

Dropped: SambaNova — free tier gates inference behind a payment method (fails the
no-billing rule). `bench()` probes every seat at startup — the agent only routes
to seats that actually answer. Seats are disposable by design.

## Kill test (verified 2026-10-10)

Full run log: [evidence/kill-test-2026-10-10.txt](evidence/kill-test-2026-10-10.txt)

```
$ RELAYZERO_KILL_SEAT=groq python relay.py "Write a haiku about failing over…"
  [!] groq failed (500); cooling down, failing over…
  [✓] answered by gemini (7115.3 ms, cost $0.00)
  verifier score: 10 (seat: openrouter)
  failovers: 1 | cost: $0.00
```

The agent did not restart — it restored state from the pre-call checkpoint,
failed over to the next live free seat, verified the answer with a second seat,
and delivered.

## Sponsor evidence (WIP — building in public for #neatHack)

- neatlogs traces: every run traced from commit #1 — [trace links land here]
- Entire checkpoints + `entire-graph` output: [lands here]
- cfo.ai business plan: [lands here] (free signup path, no card)

## The business (preview)

RelayZero as a $0-inference agent API: inference costs $0, gross margin on
inference is 100%, break-even ≈ 35 Pro customers. Full plan via cfo.ai (day 2).
