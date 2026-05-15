"""Exercise 5 - Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")


if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None

st.title("HITL PR Review Agent")


async def load_recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   MAX(CASE risk_level
                         WHEN 'low' THEN 1
                         WHEN 'med' THEN 2
                         WHEN 'high' THEN 3
                         ELSE 0
                       END) AS worst_score,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 10
            """
        ) as cur:
            rows = await cur.fetchall()
    risk_by_score = {1: "low", 2: "med", 3: "high"}
    return [
        {
            "thread_id": row["thread_id"],
            "pr_url": row["pr_url"],
            "last_event": row["last_event"],
            "worst_risk": risk_by_score.get(row["worst_score"], "unknown"),
            "events": row["events"],
        }
        for row in rows
    ]


with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = asyncio.run(load_recent_sessions())
    except Exception as exc:
        sessions = []
        st.caption(f"No sessions available yet: {exc}")

    if not sessions:
        st.caption("No audit sessions yet")

    for i, session in enumerate(sessions):
        label = f"{session['worst_risk']} risk | {session['events']} events"
        if st.button(label, key=f"session_{i}", use_container_width=True):
            st.session_state.thread_id = session["thread_id"]
            st.session_state.pr_url = session["pr_url"]
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.rerun()
        st.caption(session["pr_url"])
        st.caption(session["last_event"])


with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


def render_approval_card(payload: dict) -> dict | None:
    """58-72% bucket: show the LLM review + 3 buttons."""
    confidence = payload["confidence"]
    st.subheader(f"Approval requested - confidence {confidence:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** "
            f"`{comment['file']}:{comment.get('line') or '?'}` - {comment['body']}"
        )

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """<58% bucket: show risk factors + reviewer question form."""
    confidence = payload["confidence"]
    st.subheader(f"Strong escalation - confidence {confidence:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, question in enumerate(payload["questions"]):
            answers[question] = st.text_area(question, key=f"escalation_answer_{i}")
        submitted_answers = st.form_submit_button("Submit answers", type="primary")
        if submitted_answers:
            return answers
    return None


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or an interrupt result."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR and asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result


payload = st.session_state.interrupt_payload
if payload is not None:
    if payload["kind"] == "approval_request":
        answer = render_approval_card(payload)
    else:
        answer = render_escalation_card(payload)

    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url,
                st.session_state.thread_id,
                resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()


if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if "commit_failed" in action:
        st.error(f"{action} - GitHub comment was not posted")
    elif action.startswith("auto") or action.startswith("committed"):
        st.success(f"{action} - comment posted")
        st.link_button("View PR on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(
        f"thread_id = {st.session_state.thread_id} | "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
