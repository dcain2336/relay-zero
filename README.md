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

| Seat | Provider | Endpoint | Free model (verified 2026-10-11) |
|------|----------|----------|----------------------------------|
| groq | Groq | api.groq.com/openai/v1 | qwen/qwen3.8-27b |
| orcarouter-deepseek | OrcaRouter | api.orcarouter.ai/v1 | deepseek/deepseek-v4-flash-free |
| orcarouter-hy3 | OrcaRouter | api.orcarouter.ai/v1 | tencent/hy3-free |
| orcarouter-glm | OrcaRouter | api.orcarouter.ai/v1 | z-ai/glm-5.3-flash-free |
| openrouter | OpenRouter | openrouter.ai/api/v1 | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free (flapping 10-11: free-pool saturation) |
| gemini | Google AI | generativelanguage.googleapis.com | gemini-flash-latest (flapping 10-11: 503 demand spikes) |
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

## Kill test, day 2 (verified 2026-10-11 — run `093a4e49`)

Full run log: [evidence/kill-test-2026-10-11.txt](evidence/kill-test-2026-10-11.txt)

```
$ RELAYZERO_KILL_SEAT=groq python relay.py "Write a haiku about failing over…"
  [bench] groq: LIVE (kill-switch ARMED — will 500 mid-run)
  [bench] openrouter: dead — 502 (free pool saturated)
  [bench] gemini: dead — 503 (demand spike)
  [bench] orcarouter-deepseek: LIVE … orcarouter-hy3: LIVE … orcarouter-glm: LIVE
  [!] groq failed (500); cooling down, failing over…
  [✓] answered by orcarouter-deepseek (4113.3 ms, cost $0.00)
  verifier score: rubric pass (seat: orcarouter-glm)
  failovers: 1 | cost: $0.00
```

Day 2 was a live demonstration of *why* the bench exists: Groq, OpenRouter, and
Gemini were all flapping or dead at bench time. The agent probed, skipped the
dead seats, killed its own primary mid-run, and still finished — on three
OrcaRouter free seats added the same morning. Bench depth is the product.

## Before/after: v1 naive → v2 backoff+verify

Full comparison: [evidence/before-after-v1-v2.txt](evidence/before-after-v1-v2.txt)

| metric | v1 naive (`--naive`) | v2 backoff+verify (default) |
|---|---|---|
| failed calls to the dead seat | 2 (immediate same-seat retry) | 1 (30s cooldown) |
| checkpoint restore on failover | no | yes |
| answer verifier | none | second seat, 10/10 |
| inference cost | $0.00 | $0.00 |

The neatlogs traces told the story: v1's recover spans showed back-to-back 500s
on the same seat — wasted calls with zero information gain. That's what
motivated the cooldown + checkpoint-restore design in v2.

## Sponsor evidence

- **neatlogs**: every run traced from commit #1 — spans for plan/route/llm-call/
  recover/verify with latency, tokens, cost. Live dashboard links pending the
  free no-card account (in progress); meanwhile every span also buffers to
  `local_traces/` JSONL. Trace the kill test: `RELAYZERO_KILL_SEAT=groq python
  relay.py "…"` and read the recover/failover spans.
- **Entire**: the agent wraps each run in Entire's session lifecycle
  (`entire_wrap.py`) and commits its own run artifacts — every agent commit is
  a replayable checkpoint. Evidence: `evidence/entire-checkpoint-explain.txt`
  (checkpoint `01M4JBMESRM5N1J1G16DR8KE6C`), per-run transcripts in
  `evidence/runs/<run_id>/transcript.jsonl`. Day-2 checkpoints fire on agent
  commits too: `01M4NHB9KAH8CXKECAQ6C4TMNH` (planner/verifier failover fix),
  `01M4NHQN9EEQHM0GAYBGYGCV2A` (OrcaRouter bench expansion).
- **cfo.ai**: business numbers exported as CSV (`business/pricing-tiers.csv`,
  `business/scenarios.csv`, `business/unit-economics.csv`), ready for upload;
  Ari forecasts land day 2.

## The business (preview)

RelayZero as a $0-inference agent API: inference costs $0, gross margin on
inference is 100%, break-even ≈ 35 Pro customers. Full plan via cfo.ai (day 2).
