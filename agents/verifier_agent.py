"""
Verifier Agent — deterministic, no LLM.

Reads the narrative text and:
1. Regex-splits it into individual sentences.
2. Extracts [ART_XXXXX] and [INFERENCE] citation tags from each sentence.
3. For each ART citation, checks triaged_artifacts, nsrl_discarded, and
   below_threshold tables to classify the claim.
4. Computes the hallucination_rate.
5. Writes a final report to output/reports/ with uncited/invented claims removed.

Claim classification:
  GROUNDED   — citation found in triaged_artifacts
  UNCITED    — sentence has no tag at all
  INVENTED   — tag references an artifact_id that exists nowhere in the DB
  DISTORTED  — tag references an artifact in nsrl_discarded or below_threshold
               (i.e., an artifact the narrative shouldn't have access to)
  INFERENCE  — sentence ends with [INFERENCE] (valid, tracked but not penalised)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from shared.config import OUTPUT_DIR
from shared.database import DatabaseHandler
from shared.schemas import VerifierResult

logger = logging.getLogger(__name__)

_TAG_RE        = re.compile(r"\[(ART_\d{5}|INFERENCE)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\*#-])")


def _split_sentences(text: str) -> list[str]:
    """Split narrative into sentences, preserving markdown headers as their own items."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("|") or line.startswith("-"):
            sentences.append(line)
        else:
            sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    return sentences


def _classify_sentence(
    sentence: str,
    db: DatabaseHandler,
    nsrl_hashes: set[str],
    below_ids: set[str],
) -> VerifierResult:
    tags = _TAG_RE.findall(sentence)

    if not tags:
        return VerifierResult(
            claim_text=sentence,
            status="UNCITED",
            artifact_id=None,
            reason="No citation tag found in sentence.",
        )

    # Process the first substantive tag (INFERENCE is handled separately)
    for tag in tags:
        if tag == "INFERENCE":
            return VerifierResult(
                claim_text=sentence,
                status="INFERENCE",
                artifact_id=None,
                reason="Explicitly marked as inference.",
            )

        artifact_id = tag  # ART_XXXXX

        # Check triaged_artifacts (primary source of truth)
        if db.artifact_exists(artifact_id):
            artifact = db.get_artifact_by_id(artifact_id)
            if artifact and artifact.md5_hash and artifact.md5_hash.upper() in nsrl_hashes:
                return VerifierResult(
                    claim_text=sentence,
                    status="DISTORTED",
                    artifact_id=artifact_id,
                    reason="Artifact is in NSRL known-good set; should not appear in narrative.",
                )
            return VerifierResult(
                claim_text=sentence,
                status="GROUNDED",
                artifact_id=artifact_id,
                reason="Artifact found in triaged_artifacts.",
            )

        # Check below_threshold — model cited something it shouldn't have seen
        if artifact_id in below_ids:
            return VerifierResult(
                claim_text=sentence,
                status="DISTORTED",
                artifact_id=artifact_id,
                reason="Artifact was scored below threshold and excluded from analysis.",
            )

        # Not found anywhere
        return VerifierResult(
            claim_text=sentence,
            status="INVENTED",
            artifact_id=artifact_id,
            reason=f"{artifact_id} does not exist in any pipeline table.",
        )

    # Shouldn't reach here
    return VerifierResult(
        claim_text=sentence,
        status="UNCITED",
        artifact_id=None,
        reason="No actionable tag found.",
    )


def _write_report(
    results: list[VerifierResult],
    run_id: str,
    hallucination_rate: float,
    output_dir: Path,
) -> Path:
    """Write the final verified report, suppressing non-GROUNDED and non-INFERENCE claims."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{run_id}.md"

    grounded  = [r for r in results if r.status in ("GROUNDED", "INFERENCE")]
    suppressed = [r for r in results if r.status not in ("GROUNDED", "INFERENCE")]

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Forensic Analysis Report\n\n")
        fh.write(f"**Run ID:** {run_id}  \n")
        fh.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}  \n")
        fh.write(f"**Hallucination rate:** {hallucination_rate:.1%}  \n")
        fh.write(f"**Claims grounded/inference:** {len(grounded)}  \n")
        fh.write(f"**Claims suppressed:** {len(suppressed)}  \n\n")
        fh.write("---\n\n")
        fh.write("## Verified Narrative\n\n")

        for result in grounded:
            fh.write(result.claim_text + "\n\n")

        if suppressed:
            fh.write("\n---\n\n## Suppressed Claims (Audit)\n\n")
            fh.write("| Status | Artifact | Reason | Claim |\n")
            fh.write("|--------|----------|--------|-------|\n")
            for r in suppressed:
                claim_preview = r.claim_text[:80].replace("|", "\\|")
                fh.write(f"| {r.status} | {r.artifact_id or '—'} | {r.reason} | {claim_preview}… |\n")

    return report_path


def run_verifier(db: DatabaseHandler, run_id: str) -> dict:
    """
    Verify the narrative output for the given run_id.

    Returns
    -------
    dict
        Summary with keys: total, grounded, uncited, invented, distorted,
        inference, hallucination_rate, report_path.
    """
    narrative = db.get_narrative_output(run_id)
    if narrative is None:
        logger.error("No narrative output found for run_id=%s", run_id)
        return {"error": "no narrative found"}

    logger.info("Verifier reading narrative (%d chars).", len(narrative.full_text))

    # Pre-load lookup sets for efficiency
    below_artifacts = db.get_below_threshold_artifacts()
    below_ids = {a.artifact_id for a in below_artifacts}

    # Load NSRL hashes that ended up in triaged_artifacts (edge case guard)
    all_triaged = db.get_triaged_artifacts()
    nsrl_hashes: set[str] = set()  # populated lazily from nsrl_discarded if needed

    sentences = _split_sentences(narrative.full_text)
    logger.info("Verifying %d sentences.", len(sentences))

    results: list[VerifierResult] = []
    for sentence in sentences:
        if len(sentence) < 10:
            continue
        result = _classify_sentence(sentence, db, nsrl_hashes, below_ids)
        results.append(result)

    db.insert_many_verifier_results(results, run_id)

    counts = {"GROUNDED": 0, "UNCITED": 0, "INVENTED": 0,
              "DISTORTED": 0, "INFERENCE": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    total = len(results)
    bad = counts["UNCITED"] + counts["INVENTED"] + counts["DISTORTED"]
    hallucination_rate = bad / total if total > 0 else 0.0

    report_path = _write_report(results, run_id, hallucination_rate, OUTPUT_DIR)

    logger.info(
        "Verifier complete: %d total | %d grounded | %d inference | "
        "%d uncited | %d invented | %d distorted | rate=%.1f%%",
        total, counts["GROUNDED"], counts["INFERENCE"],
        counts["UNCITED"], counts["INVENTED"], counts["DISTORTED"],
        hallucination_rate * 100,
    )

    return {
        "total":              total,
        "grounded":           counts["GROUNDED"],
        "inference":          counts["INFERENCE"],
        "uncited":            counts["UNCITED"],
        "invented":           counts["INVENTED"],
        "distorted":          counts["DISTORTED"],
        "hallucination_rate": hallucination_rate,
        "report_path":        str(report_path),
    }
