"""
LangGraph orchestrator for the forensic analysis pipeline.

State machine:
  START → triage_node → timeline_node → ioc_node → narrative_node
        → verifier_node → END

  Any node failure routes to error_node which writes diagnostics and halts.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Annotated, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.ioc_agent import run_ioc
from agents.narrative_agent import run_narrative
from agents.timeline_agent import run_timeline
from agents.triage_agent import run_triage
from agents.verifier_agent import run_verifier
from shared.config import DATABASE_PATH, NSRL_HASH_PATH, NSRL_INDEX_PATH, OLLAMA_MODEL
from shared.database import DatabaseHandler
from shared.llm_client import LLMClient
from shared.nsrl_loader import NSRLLoader

logger = logging.getLogger(__name__)

# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    run_id: str
    current_agent: str
    error_state: Optional[str]
    # Per-agent result counts (for logging / evaluation)
    triage_count: int
    timeline_count: int
    ioc_count: int
    narrative_text: str
    verifier_summary: dict


# ── Shared resources (initialised once per pipeline run) ─────────────────────

def _build_resources(run_id: str) -> tuple[DatabaseHandler, NSRLLoader, LLMClient]:
    db    = DatabaseHandler(DATABASE_PATH)
    nsrl  = NSRLLoader(NSRL_HASH_PATH, NSRL_INDEX_PATH)
    llm   = LLMClient(db=db, run_id=run_id, agent_name="pipeline", model=OLLAMA_MODEL)
    return db, nsrl, llm


# ── Node implementations ──────────────────────────────────────────────────────

def triage_node(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db, nsrl, _ = _build_resources(run_id)
    logger.info("=== TRIAGE NODE (run_id=%s) ===", run_id)
    try:
        count = run_triage(db, nsrl)
        return {**state, "current_agent": "triage", "triage_count": count}
    except Exception as exc:
        logger.error("Triage failed: %s", exc, exc_info=True)
        return {**state, "current_agent": "triage", "error_state": str(exc)}


def timeline_node(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db, _, _ = _build_resources(run_id)
    logger.info("=== TIMELINE NODE (run_id=%s) ===", run_id)
    try:
        count = run_timeline(db)
        return {**state, "current_agent": "timeline", "timeline_count": count}
    except Exception as exc:
        logger.error("Timeline failed: %s", exc, exc_info=True)
        return {**state, "current_agent": "timeline", "error_state": str(exc)}


def ioc_node(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db, _, llm = _build_resources(run_id)
    llm.agent_name = "ioc_agent"
    logger.info("=== IOC NODE (run_id=%s) ===", run_id)
    try:
        count = run_ioc(db, llm=llm)
        return {**state, "current_agent": "ioc", "ioc_count": count}
    except Exception as exc:
        logger.error("IOC extraction failed: %s", exc, exc_info=True)
        return {**state, "current_agent": "ioc", "error_state": str(exc)}


def narrative_node(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db, _, llm = _build_resources(run_id)
    llm.agent_name = "narrative_agent"
    logger.info("=== NARRATIVE NODE (run_id=%s) ===", run_id)
    try:
        text = run_narrative(db, llm, run_id)
        return {**state, "current_agent": "narrative", "narrative_text": text}
    except Exception as exc:
        logger.error("Narrative generation failed: %s", exc, exc_info=True)
        return {**state, "current_agent": "narrative", "error_state": str(exc)}


def verifier_node(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db, _, _ = _build_resources(run_id)
    logger.info("=== VERIFIER NODE (run_id=%s) ===", run_id)
    try:
        summary = run_verifier(db, run_id)
        return {**state, "current_agent": "verifier", "verifier_summary": summary}
    except Exception as exc:
        logger.error("Verifier failed: %s", exc, exc_info=True)
        return {**state, "current_agent": "verifier", "error_state": str(exc)}


def error_node(state: PipelineState) -> PipelineState:
    logger.error(
        "Pipeline halted at agent='%s' with error: %s",
        state.get("current_agent"), state.get("error_state"),
    )
    return {**state, "current_agent": "error"}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after(state: PipelineState, next_node: str) -> str:
    if state.get("error_state"):
        return "error_node"
    return next_node


def route_triage(state: PipelineState) -> str:
    return _route_after(state, "timeline_node")

def route_timeline(state: PipelineState) -> str:
    return _route_after(state, "ioc_node")

def route_ioc(state: PipelineState) -> str:
    return _route_after(state, "narrative_node")

def route_narrative(state: PipelineState) -> str:
    return _route_after(state, "verifier_node")

def route_verifier(state: PipelineState) -> str:
    return END

def route_error(state: PipelineState) -> str:
    return END


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("triage_node",    triage_node)
    graph.add_node("timeline_node",  timeline_node)
    graph.add_node("ioc_node",       ioc_node)
    graph.add_node("narrative_node", narrative_node)
    graph.add_node("verifier_node",  verifier_node)
    graph.add_node("error_node",     error_node)

    graph.add_edge(START, "triage_node")

    graph.add_conditional_edges("triage_node",   route_triage,
                                 {"timeline_node": "timeline_node", "error_node": "error_node"})
    graph.add_conditional_edges("timeline_node", route_timeline,
                                 {"ioc_node": "ioc_node", "error_node": "error_node"})
    graph.add_conditional_edges("ioc_node",      route_ioc,
                                 {"narrative_node": "narrative_node", "error_node": "error_node"})
    graph.add_conditional_edges("narrative_node", route_narrative,
                                 {"verifier_node": "verifier_node", "error_node": "error_node"})
    graph.add_conditional_edges("verifier_node", route_verifier, {END: END})
    graph.add_conditional_edges("error_node",    route_error,    {END: END})

    return graph


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    run_id = str(uuid.uuid4())
    logger.info("Starting pipeline run_id=%s at %s", run_id,
                datetime.utcnow().isoformat())

    initial_state: PipelineState = {
        "run_id":           run_id,
        "current_agent":    "init",
        "error_state":      None,
        "triage_count":     0,
        "timeline_count":   0,
        "ioc_count":        0,
        "narrative_text":   "",
        "verifier_summary": {},
    }

    graph    = build_graph()
    compiled = graph.compile()
    final    = compiled.invoke(initial_state)

    if final.get("error_state"):
        logger.error("Pipeline finished with error: %s", final["error_state"])
    else:
        summary = final.get("verifier_summary", {})
        logger.info(
            "Pipeline complete | triage=%d timeline=%d ioc=%d "
            "hallucination_rate=%.1f%% report=%s",
            final["triage_count"], final["timeline_count"], final["ioc_count"],
            summary.get("hallucination_rate", 0) * 100,
            summary.get("report_path", "n/a"),
        )

    return final


if __name__ == "__main__":
    run_pipeline()
