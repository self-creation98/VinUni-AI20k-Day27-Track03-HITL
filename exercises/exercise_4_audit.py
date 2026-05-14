"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

TODOs to complete (10 total): the audit() body, ONE AuditEntry per node for
analyze/route/commit/auto_approve/synthesize, TWO each for human_approval and
escalate (before + after the interrupt).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    # TODO: call write_audit_event(thread_id=state["thread_id"],
    #                              pr_url=state["pr_url"], entry=entry)
    raise NotImplementedError("Implement the audit() body — one call to write_audit_event")


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Senior reviewer. Structured output."},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    # TODO: build and emit an AuditEntry for this step. Use the LLM's output `a`.
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    # TODO: emit an AuditEntry — this is the first row where `decision` is real.
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    # TODO #1 — audit BEFORE the interrupt. No human has responded yet.

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    # TODO #2 — audit AFTER resume. Now you know the reviewer's choice
    #           (approve / reject / edit), their feedback, and who they are
    #           (os.environ.get("GITHUB_USER")).
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


async def node_commit(state):
    t0 = time.monotonic()
    action = "committed" if state.get("human_choice") == "approve" else "rejected"
    # TODO: emit an AuditEntry summarising what was committed (or rejected).
    return {"final_action": action}


async def node_auto_approve(state):
    t0 = time.monotonic()
    a = state["analysis"]
    # TODO: emit an AuditEntry — no human was involved, decision is "auto".
    return {"final_action": "auto_approved"}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    # TODO #1 — audit BEFORE the interrupt (reviewer hasn't answered yet).

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    # TODO #2 — audit AFTER resume. You now have the answers.
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Refine review with reviewer answers."},
            {"role": "user", "content": f"Diff:\n{state['pr_diff']}\n\nQ&A:\n{qa}"},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    # TODO: emit an AuditEntry — use the NEW confidence (refined.confidence),
    #       which should be higher than the original analysis.
    return {"analysis": refined, "final_action": "escalated_then_synthesized"}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
