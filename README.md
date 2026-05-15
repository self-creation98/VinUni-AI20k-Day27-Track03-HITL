# HITL PR Review Agent

This project implements a human-in-the-loop pull-request review agent with
LangGraph, GitHub REST API integration, Streamlit approval UI, and a structured
SQLite audit trail.

The agent reads a GitHub pull request, analyzes the diff with an LLM, routes the
review by confidence, optionally asks for human input, posts a review comment,
and records every meaningful event in `hitl_audit.db`.

## Features

- Fetches pull request metadata and diff from GitHub.
- Produces structured LLM review output with summary, risk factors, comments,
  confidence, confidence reasoning, and escalation questions.
- Routes by confidence:
  - `>= 0.73`: auto approval
  - `0.58 - 0.72`: human approval
  - `< 0.58`: escalation with reviewer Q&A
- Supports LangGraph `interrupt()` and resume with `Command(resume=...)`.
- Provides a Streamlit UI for PR review approval and escalation.
- Persists graph checkpoints with `AsyncSqliteSaver`.
- Writes structured audit events to SQLite for replay/debugging.

## Project Structure

```text
.
|-- app.py                         # Streamlit UI
|-- common/
|   |-- db.py                      # SQLite audit helpers
|   |-- github.py                  # GitHub REST API client
|   |-- llm.py                     # OpenRouter/OpenAI-compatible LLM factory
|   `-- schemas.py                 # Graph state + Pydantic models
|-- exercises/
|   |-- exercise_1_confidence.py   # Confidence routing graph
|   |-- exercise_2_hitl.py         # HITL approval with interrupt/resume
|   |-- exercise_3_escalation.py   # Escalation Q&A and synthesis
|   `-- exercise_4_audit.py        # Durable graph + SQLite audit trail
`-- audit/
    |-- schema.sql                 # audit_events schema
    `-- replay.py                  # Audit replay CLI
```

## Requirements

- Python 3.11+
- OpenRouter API key
- GitHub Personal Access Token with `public_repo` scope

The project was tested locally with `python -m streamlit run app.py`. If `uv` is
available, the commands can also be run with `uv run`.

## Environment Setup

Create a `.env` file at the repo root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key
LLM_MODEL=openai/gpt-4o-mini
LLM_BASE_URL=https://openrouter.ai/api/v1
GITHUB_TOKEN=ghp_your-token
```

Do not commit `.env`. It contains secrets and is ignored by `.gitignore`.

## Run the Streamlit App

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Enter one of the demo pull request URLs:

```text
https://github.com/VinUni-AI20k/PR-Demo/pull/1
https://github.com/VinUni-AI20k/PR-Demo/pull/2
```

Expected behavior:

| PR | Expected route | What happens |
| --- | --- | --- |
| PR #1 | `human_approval` | Shows approval UI with Approve / Reject / Edit |
| PR #2 | `escalate` | Shows escalation questions, then synthesizes a refined review |

Approving or completing escalation can post a real comment to GitHub using the
configured `GITHUB_TOKEN`.

## Run Individual Exercises

Exercise 1:

```powershell
python exercises\exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
python exercises\exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
```

Exercise 2:

```powershell
python exercises\exercise_2_hitl.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
```

Exercise 3:

```powershell
python exercises\exercise_3_escalation.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
```

Exercise 4:

```powershell
python exercises\exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
```

## Audit Trail

The app creates `hitl_audit.db` on first run. It contains:

- `audit_events`: structured decision/event log
- `checkpoints`: LangGraph checkpoint state
- `writes`: LangGraph checkpoint writes

List recent review sessions:

```powershell
python -m audit.replay --list
```

Replay one session:

```powershell
python -m audit.replay --thread <thread_id>
```

Each audit event records:

- timestamp
- thread ID
- PR URL
- agent ID
- action
- confidence
- risk level
- reviewer ID
- decision
- reason
- execution time

## Verification

Run a syntax check:

```powershell
python -m compileall common audit exercises app.py
```

Check Git status before submission:

```powershell
git status --short
```

Files that should not be submitted:

- `.env`
- local virtual environments
- Python caches

`hitl_audit.db` is ignored by default because it is a local runtime artifact. If
the instructor asks for audit evidence, provide the database file separately or
share the `thread_id` plus replay output.
