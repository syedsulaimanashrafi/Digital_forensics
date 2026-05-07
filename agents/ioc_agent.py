"""
IOC Agent — regex extraction + optional LLM classification for ambiguous items.

Pass 1: deterministic regex over all triaged artifact fields (name, full_path,
        raw_value) to extract IPs, emails, MD5/SHA hashes, MAC addresses, domains.
Pass 2: ambiguous items (bare usernames, unusual file paths in raw_value) are
        batched and sent to the LLM for classification.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from shared.database import DatabaseHandler
from shared.llm_client import LLMClient
from shared.schemas import IOCEntity, TriagedArtifact

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_RE_MD5 = re.compile(r"\b[0-9a-fA-F]{32}\b")
_RE_SHA1 = re.compile(r"\b[0-9a-fA-F]{40}\b")
_RE_SHA256 = re.compile(r"\b[0-9a-fA-F]{64}\b")
_RE_MAC = re.compile(r"\b([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")
_RE_DOMAIN = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"(?:com|net|org|edu|gov|io|co|info|biz|[a-z]{2})\b",
    re.IGNORECASE,
)
# Username heuristic: short word after \Users\ or in raw_value starting with letters
_RE_USERNAME = re.compile(r"\\Users\\([A-Za-z][A-Za-z0-9._\-]{1,30})\\")


def _extract_from_text(
    text: str, artifact_id: str
) -> list[IOCEntity]:
    """Run all regex patterns against a single text blob."""
    entities: list[IOCEntity] = []

    def add(entity_type: str, value: str, confidence: float = 1.0,
            classification: Optional[str] = None) -> None:
        entities.append(IOCEntity(
            artifact_id=artifact_id,
            entity_type=entity_type,
            value=value.strip(),
            classification=classification,
            confidence=confidence,
        ))

    for m in _RE_IPV4.finditer(text):
        ip = m.group()
        # Exclude loopback and link-local
        if not (ip.startswith("127.") or ip.startswith("169.254.")):
            add("IP", ip)

    for m in _RE_EMAIL.finditer(text):
        add("EMAIL", m.group())

    for m in _RE_SHA256.finditer(text):
        add("HASH", m.group(), classification="SHA256")
    for m in _RE_SHA1.finditer(text):
        add("HASH", m.group(), classification="SHA1")
    for m in _RE_MD5.finditer(text):
        # Avoid re-tagging something already caught as SHA1/SHA256 overlap
        add("HASH", m.group(), classification="MD5")

    for m in _RE_MAC.finditer(text):
        add("MAC", m.group())

    for m in _RE_DOMAIN.finditer(text):
        dom = m.group()
        # Skip domains that are part of an already-extracted email
        if "@" not in text[max(0, m.start() - 60):m.start()]:
            add("DOMAIN", dom)

    for m in _RE_USERNAME.finditer(text):
        add("USERNAME", m.group(1), confidence=0.9)

    return entities


def _all_text(artifact: TriagedArtifact) -> str:
    parts = [artifact.name, artifact.full_path, artifact.raw_value or ""]
    return " ".join(parts)


# ── LLM disambiguation ────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a digital forensics analyst. Classify each item as one of: "
    "IP, EMAIL, HASH, DOMAIN, USERNAME, PATH, OTHER. "
    "Return JSON: {\"classifications\": [{\"value\": \"...\", \"type\": \"...\"}]}"
)

_AMBIGUOUS_SOURCE_MODULES = {
    "web_history", "web_search", "web_bookmarks",
    "recent_documents", "clipboard",
}


def _needs_llm_classification(artifact: TriagedArtifact) -> bool:
    return artifact.source_module.lower() in _AMBIGUOUS_SOURCE_MODULES


# ── Main agent function ───────────────────────────────────────────────────────

def run_ioc(
    db: DatabaseHandler,
    llm: Optional[LLMClient] = None,
    batch_size: int = 30,
) -> int:
    """
    Extract IOC entities from all triaged artifacts.

    Parameters
    ----------
    db : DatabaseHandler
    llm : LLMClient | None
        If None, the LLM pass is skipped (regex-only mode).
    batch_size : int
        How many ambiguous items to send per LLM call.

    Returns
    -------
    int
        Total number of IOC entities written to ioc_entities.
    """
    artifacts = db.get_triaged_artifacts()
    if not artifacts:
        logger.warning("No triaged artifacts found — IOC extraction skipped.")
        return 0

    logger.info("Extracting IOCs from %d artifacts (regex pass).", len(artifacts))

    all_entities: list[IOCEntity] = []
    ambiguous_batches: list[list[TriagedArtifact]] = []
    current_batch: list[TriagedArtifact] = []

    # Pass 1 — regex
    for artifact in artifacts:
        text = _all_text(artifact)
        entities = _extract_from_text(text, artifact.artifact_id)
        all_entities.extend(entities)

        if llm and _needs_llm_classification(artifact):
            current_batch.append(artifact)
            if len(current_batch) >= batch_size:
                ambiguous_batches.append(current_batch)
                current_batch = []

    if current_batch:
        ambiguous_batches.append(current_batch)

    # Pass 2 — LLM (only if client provided)
    if llm and ambiguous_batches:
        logger.info("LLM pass: %d batch(es) of ambiguous items.", len(ambiguous_batches))
        for batch in ambiguous_batches:
            items_text = "\n".join(
                f"- artifact_id={a.artifact_id} raw_value={a.raw_value or a.name}"
                for a in batch
            )
            prompt = (
                f"Classify these forensic artifact values as IOC entities:\n{items_text}\n"
                "Return JSON with a 'classifications' array."
            )
            try:
                import json
                response = llm.chat(prompt, system=_LLM_SYSTEM)
                parsed = json.loads(response)
                for item in parsed.get("classifications", []):
                    # Match back to artifact_id via value
                    for artifact in batch:
                        if item.get("value") in _all_text(artifact):
                            all_entities.append(IOCEntity(
                                artifact_id=artifact.artifact_id,
                                entity_type=item.get("type", "OTHER"),
                                value=item.get("value", ""),
                                classification="LLM",
                                confidence=0.75,
                            ))
                            break
            except Exception as exc:
                logger.warning("LLM IOC batch failed: %s", exc)

    # Deduplicate by (artifact_id, entity_type, value)
    seen: set[tuple[str, str, str]] = set()
    unique: list[IOCEntity] = []
    for e in all_entities:
        key = (e.artifact_id, e.entity_type, e.value)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    if unique:
        db.insert_many_ioc_entities(unique)

    logger.info("IOC extraction complete: %d unique entities.", len(unique))
    return len(unique)
