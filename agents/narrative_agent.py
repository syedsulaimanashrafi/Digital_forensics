"""
Narrative Agent — LLM via Ollama, structured output via Instructor.

Reads timeline_events (top 30), ioc_entities, and OS info from triaged
artifacts, renders the Jinja2 template, and calls the LLM with Instructor
schema enforcement to produce a cited forensic narrative.

Every claim must end with [ART_XXXXX] or [INFERENCE]; the verifier agent
checks these citations downstream.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel, Field

from shared.config import PROMPT_VERSION, PROMPTS_DIR
from shared.database import DatabaseHandler
from shared.llm_client import LLMClient, _check_ollama
from shared.schemas import IOCEntity, NarrativeOutput, TimelineEvent

logger = logging.getLogger(__name__)

# ── Response schema for Instructor ───────────────────────────────────────────

class NarrativeResponse(BaseModel):
    """Schema the LLM must return via Instructor."""
    narrative_text: str = Field(
        description=(
            "Full forensic narrative. Every factual sentence ends with "
            "[ART_XXXXX] or [INFERENCE]. Approx 500-700 words."
        )
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"\[ART_\d{5}\]|\[INFERENCE\]")


def _count_claims(text: str) -> int:
    """Count sentences that contain a citation tag."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sum(1 for s in sentences if _TAG_RE.search(s))


def _infer_os_info(db: DatabaseHandler) -> Optional[str]:
    """Try to extract an OS identifier from artifact names/paths."""
    artifacts = db.get_triaged_artifacts()
    for a in artifacts:
        if "windows" in a.full_path.lower() or "ntfs" in a.source_module.lower():
            return "Windows (inferred from artifact paths)"
        if "system32" in a.full_path.lower():
            return "Windows (System32 path detected)"
    return None


def _render_prompt(
    timeline_events: list[TimelineEvent],
    ioc_entities: list[IOCEntity],
    artifact_count: int,
    os_info: Optional[str],
) -> str:
    try:
        env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), autoescape=False)
        template = env.get_template(f"{PROMPT_VERSION}.j2")
    except TemplateNotFound:
        raise FileNotFoundError(
            f"Prompt template '{PROMPT_VERSION}.j2' not found in {PROMPTS_DIR}"
        )

    return template.render(
        timeline_events=timeline_events,
        timeline_count=len(timeline_events),
        ioc_entities=ioc_entities,
        ioc_count=len(ioc_entities),
        artifact_count=artifact_count,
        os_info=os_info,
    )


# ── Main agent function ───────────────────────────────────────────────────────

def run_narrative(db: DatabaseHandler, llm: LLMClient, run_id: str) -> str:
    """
    Generate the forensic narrative and persist it to narrative_output.

    Returns
    -------
    str
        The narrative text.

    Raises
    ------
    RuntimeError
        If Ollama is not reachable before the call is attempted.
    """
    if not _check_ollama():
        raise RuntimeError(
            "Ollama is not reachable. Start the Ollama server before running "
            "the narrative agent."
        )

    timeline_events = db.get_timeline_events(limit=30)
    ioc_entities    = db.get_ioc_entities()
    artifact_count  = db.count_triaged_artifacts()
    os_info         = _infer_os_info(db)

    logger.info(
        "Narrative agent: %d timeline events, %d IOCs, %d triaged artifacts.",
        len(timeline_events), len(ioc_entities), artifact_count,
    )

    prompt = _render_prompt(timeline_events, ioc_entities, artifact_count, os_info)

    logger.info("Calling LLM for narrative generation…")
    response: NarrativeResponse = llm.chat_structured(
        prompt=prompt,
        response_model=NarrativeResponse,
        max_retries=1,
    )

    narrative_text = response.narrative_text
    claim_count = _count_claims(narrative_text)

    db.insert_narrative_output(run_id, narrative_text, claim_count)
    logger.info(
        "Narrative complete: %d claimed sentences, %d citation tags found.",
        claim_count, len(_TAG_RE.findall(narrative_text)),
    )

    return narrative_text
