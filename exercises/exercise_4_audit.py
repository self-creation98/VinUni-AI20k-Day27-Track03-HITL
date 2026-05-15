"""Exercise 4 - Structured SQLite audit trail + durable checkpointer."""

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
from common.github import fetch_pr, post_review_comment
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
HIGH_RISK_TERMS = (
    "auth",
    "password",
    "token",
    "md5",
    "sql injection",
    "login",
    "cloud sync",
    "plaintext",
    "hard-coded",
    "security",
)


async def audit(state: ReviewState, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the audit_events table."""
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


def reviewer_id() -> str | None:
    """Best-effort local reviewer identity for HITL audit rows."""
    return (
        os.environ.get("GITHUB_USER")
        or os.environ.get("GITHUB_ACTOR")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
    )


def calibrate_analysis(state: ReviewState, analysis: PRAnalysis) -> PRAnalysis:
    """Keep small non-security PRs in HITL approval while preserving true escalations."""
    review_text = " ".join([
        analysis.summary,
        analysis.confidence_reasoning,
        *analysis.risk_factors,
        *(comment.body for comment in analysis.comments),
        state["pr_diff"][:4000],
    ]).lower()
    has_high_risk_signal = any(term in review_text for term in HIGH_RISK_TERMS)
    if analysis.confidence < ESCALATE_THRESHOLD and not has_high_risk_signal:
        return analysis.model_copy(update={
            "confidence": 0.65,
            "confidence_reasoning": (
                f"{analysis.confidence_reasoning} Calibrated as medium confidence: "
                "the remaining concerns are reviewer-confirmable and no auth, token, "
                "SQL injection, password, or security-sensitive changes were detected."
            ),
        })
    return analysis


async def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
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


async def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "Senior reviewer. Structured output. If confidence is below 60%, "
                "populate escalation_questions with 2-4 specific, context-rich "
                "questions for the reviewer."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    analysis = calibrate_analysis(state, analysis)
    console.print(f"  [green]OK[/green] confidence={analysis.confidence:.0%}, {len(analysis.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        decision="pending",
        reason=analysis.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": analysis}


async def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    t0 = time.monotonic()
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
        audit_decision = "auto"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
        audit_decision = "escalate"
    else:
        decision = "human_approval"
        audit_decision = "pending"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={confidence:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=confidence,
        risk_level=risk_level_for(confidence),
        decision=audit_decision,
        reason=f"Routed to {decision} using configured confidence thresholds",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_human_approval(state: ReviewState) -> dict:
    t0 = time.monotonic()
    analysis = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        decision="pending",
        reason="Awaiting reviewer approve/reject/edit decision",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "comments": [c.model_dump() for c in analysis.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        reviewer_id=reviewer_id(),
        decision=response.get("choice", "pending"),
        reason=response.get("feedback") or "Reviewer approved without feedback",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def _render_comment_body(state: ReviewState) -> str:
    analysis = state["analysis"]
    lines = [f"### Automated review (confidence {analysis.confidence:.0%})", "", analysis.summary, ""]
    for comment in analysis.comments:
        lines.append(f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` - {comment.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for question, answer in state["escalation_answers"].items():
            lines.append(f"> **{question}** {answer}")
    return "\n".join(lines)


def _post(state: ReviewState) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as exc:
        console.print(f"  [red]FAIL[/red] post failed: {exc}")
        return "commit_failed"


async def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    t0 = time.monotonic()
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        console.print(f"  [yellow]*[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"

    analysis = state["analysis"]
    if action == "committed" and state.get("escalation_answers"):
        decision = "escalate"
    elif action == "committed":
        decision = "approve"
    elif action == "rejected":
        decision = state.get("human_choice") or "reject"
    else:
        decision = "pending"

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        reviewer_id=reviewer_id() if state.get("human_choice") or state.get("escalation_answers") else None,
        decision=decision,
        reason=f"Final action: {action}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


async def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan]  [dim]high confidence - posting directly[/dim]")
    t0 = time.monotonic()
    analysis = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        decision="auto",
        reason=f"High-confidence auto approval; final action: {action}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state: ReviewState) -> dict:
    t0 = time.monotonic()
    analysis = state["analysis"]
    questions = analysis.escalation_questions or [
        "What is the intent of this PR?",
        "Are there any migration, security, or production rollout concerns?",
    ]

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        decision="pending",
        reason="Awaiting reviewer answers: " + " | ".join(questions),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "risk_factors": analysis.risk_factors,
        "questions": questions,
    })

    answer_summary = " | ".join(f"{question}: {answer}" for question, answer in answers.items())
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        reviewer_id=reviewer_id(),
        decision="escalate",
        reason=answer_summary,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state: ReviewState) -> dict:
    console.print("[cyan]-> synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    initial = state["analysis"].model_dump_json(indent=2)
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "Refine the PR review with reviewer answers. Return a complete "
                "structured review, including updated comments and confidence."
            )},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n\n"
                f"Original analysis:\n{initial}\n\n"
                f"Reviewer Q&A:\n{qa}\n\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        reviewer_id=reviewer_id(),
        decision="escalate",
        reason=refined.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload: dict) -> dict:
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


async def run(pr_url: str, thread_id: str | None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--thread", help="Resume an existing thread")
    args = parser.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
