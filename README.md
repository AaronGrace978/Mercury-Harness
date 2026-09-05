# Mercury Harness

![Mercury Harness](docs/mercury-harness-readme.png)

A premium agent harness that **embeds how a frontier model operates into lesser models**.

Start the work with a frontier model. Mercury records the *procedure* — tool order, recovery loops, what not to touch — and injects that operational knowledge into the next session even if that session runs on a cheaper model.

This is not answer distillation and not document RAG. Lesser models already have facts. What they lack is the frontier model's **operating system**: search before edit, read the failing test, recover from a red command without repeating the same patch.

```
Frontier run                     Mercury                         Lesser run
─────────────                    ───────                         ──────────
search → read →                  distill HOW                     operating pack
test fails →                     into cards                      is prepended
patch the real site →            embed + store                   so the student
retest passes                    contrast vs fails               opens like Opus
```

## Why this works

A frontier coding agent and a small model often share a tool belt. They diverge on **policy**:

| Frontier (teacher) | Lesser (student) |
|---|---|
| Grep / read first | Patch the first file that sounds related |
| Reproduce the failure | Assume the diagnosis |
| Recover by changing approach | Retry the same edit |
| Touch `session.ts` | Rewrite `login.tsx` |

Mercury captures those divergences as **operational cards**, embeds them in a local store, and compiles a **Frontier Operating Pack** that fits the student model's context budget.

Run the same class of task later on `gpt-4o-mini` or Haiku and the pack is sitting in the system prompt: the student inherits the teacher's hands, not a copied solution.

## Flywheel

1. **Capture** a frontier agent trace (Mercury JSON, OpenAI messages, or Cursor-like tool turns).
2. **Distill** playbooks, tool policies, recovery cards, heuristics, anti-patterns.
3. **Contrast** a failed lesser run against a successful frontier run on the same task — the highest-signal cards ("don't start with `search_replace` on `login.tsx`").
4. **Embed** with a local hashing vector + BM25 hybrid retriever (no API key, no model download).
5. **Inject** a token-budgeted operating pack at lesser-model session start, or as a Cursor rule.

Teacher traces are gated by model tier. Opus / Fable / GPT-5.6 Sol / Grok-4 / o3 teach — and so do Ollama Cloud flagships such as `deepseek-v4-pro`, `kimi-k3`, `glm-5.3`, `minimax-m3`, `gpt-oss:120b`, and `qwen3.5:397b`. Mini / Haiku / Flash / Luna receive. Successful student traces do not pollute the store unless you opt in.

### Recognized model tiers

| Tier | Role | Examples |
|---|---|---|
| **Frontier** (teacher) | Distill operating packs | `claude-opus-4.1`, `claude-fable-5`, `gpt-5.6-sol`, `grok-4.5`, Ollama Cloud: `deepseek-v4-pro`, `kimi-k3`, `glm-5.3`, `minimax-m3`, `mistral-large-3`, `gpt-oss:120b`, `nemotron-3-ultra` |
| **Capable** (student) | Receive larger packs | `claude-sonnet-5`, `gpt-5.6-terra`, `gpt-oss:20b`, `gemma4:31b`, `nemotron-3-super`, `qwen3.5` |
| **Lesser** (student) | Receive tight packs | `gpt-4o-mini`, `gpt-5.6-luna`, `haiku`, `gemini-3.5-flash`, `deepseek-v4-flash`, `glm-5.3-flash`, `nemotron-3-nano` |

Cloud and local Ollama ids both work (`kimi-k3:cloud`, `gpt-oss:120b-cloud`).

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10+. Runtime dependency: Pydantic v2.

## Quick start

```bash
mercury init
mercury demo
```

`mercury demo` captures two built-in Opus traces (auth redirect + flaky pytest), contrasts them with a failed mini run, and prints the pack a lesser model would see for the login bug.

### Capture your own frontier run

```bash
mercury capture traces/opus-auth.json
mercury pack --task "Users bounce to /login after authenticating" --model gpt-4o-mini
```

Write the pack into a Cursor rule:

```bash
mercury pack --task "Fix the login redirect loop" --model gpt-4o-mini --format cursor-rule \
  > .cursor/rules/mercury.mdc
```

Export JSON for a custom agent loop:

```bash
mercury pack --task "Fix the login redirect loop" --format json
```

### Contrast a cheap failure with a frontier success

```bash
mercury contrast traces/mini-failed.json traces/opus-success.json
```

That is the core innovation: **negative knowledge**. The lesser model documents what not to do; the frontier model documents the replacement procedure.

## Python API

```python
from mercury import MercuryHarness

harness = MercuryHarness.init(".mercury")
harness.capture("traces/opus-auth.json")
pack = harness.pack(
    "Login redirects back to /login after a successful password check",
    model="gpt-4o-mini",
)
print(pack.render())
```

Drop `pack.render()` into the student system's prompt. Mid-run, if a tool comes back red, retrieve recovery cards with the error text:

```python
pack = harness.pack(
    "still bouncing to /login",
    model="gpt-4o-mini",
    error_signature="AssertionError: expected '/login' to be '/dashboard'",
)
```

## What gets distilled

| Card | Source |
|---|---|
| **Playbook** | Compressed phase sequence: explore → localize → edit → verify |
| **Tool policy** | First action + preferred tool order |
| **Recovery** | Error-like tool result + the next 1–5 frontier moves |
| **Heuristic** | Test-before-patch, retest-after-edit |
| **Anti-pattern** | Self-corrections ("actually…") and edit-first failures |
| **Standing order** | Majority behavior across frontier traces |
| **Contrast** | Student/teacher divergence on a matched task |

Cards carry confidence, task type, languages, and an optional error signature so retrieval can bias toward the current failure.

## Trace format

Canonical Mercury JSON:

```json
{
  "id": "trace_frontier_auth_redirect",
  "model": "claude-opus-4.1",
  "task": "Login keeps redirecting back to /login after a successful password check",
  "outcome": { "status": "success", "summary": "SameSite Strict → Lax" },
  "events": [
    { "type": "user", "content": "Login keeps redirecting..." },
    {
      "type": "assistant",
      "content": "Search before editing.",
      "tool_calls": [{ "name": "grep", "arguments": { "pattern": "SameSite" } }]
    },
    { "type": "tool", "tool_name": "grep", "content": "src/lib/session.ts:18: sameSite: 'strict'" }
  ]
}
```

OpenAI-style `messages` with `tool_calls` / `role: tool` are accepted. Outcome `success` | `failure` | `partial` | `unknown` gates teaching strength.

## Design choices

- **Operate, don't memorize.** The pack tells the student *how to move*, not the patch to paste. That is what transfers across files and codebases.
- **Local by default.** Embeddings are character n-gram hashing; lexical ranking is BM25. Hybrid fusion is reciprocal rank. The store is SQLite under `.mercury/`.
- **Budget by tier.** Mini-class models get ~1400 tokens of pack. Sonnet-class gets more. Frontier students get nothing — they are the teacher.
- **Teachers are gated.** A successful Haiku run does not overwrite Opus policy unless `--teacher` or `allow_student_success`.

## CLI

```
mercury init
mercury capture TRACE.json [--teacher]
mercury distill
mercury pack --task "..." [--model NAME] [--error SIG] [--format markdown|cursor-rule|json]
mercury contrast STUDENT.json TEACHER.json
mercury status
mercury demo
```

## Tests

```bash
pytest
```
