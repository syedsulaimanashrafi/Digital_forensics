"""
Timeline Agent — deterministic, no LLM.

Reads triaged_artifacts that have at least one timestamp, creates one
TimelineEvent per (artifact, timestamp-type) pair, then clusters events
that fall within a 5-minute window into the same activity burst.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from shared.database import DatabaseHandler
from shared.schemas import TimelineEvent, TriagedArtifact

logger = logging.getLogger(__name__)

_CLUSTER_WINDOW = timedelta(minutes=5)


def _events_from_artifact(artifact: TriagedArtifact) -> list[tuple[datetime, str]]:
    """Return (timestamp, event_type) pairs for every non-null timestamp."""
    pairs: list[tuple[datetime, str]] = []
    if artifact.created:
        pairs.append((artifact.created, "CREATED"))
    if artifact.modified:
        pairs.append((artifact.modified, "MODIFIED"))
    if artifact.accessed:
        pairs.append((artifact.accessed, "ACCESSED"))
    return pairs


def _build_description(artifact: TriagedArtifact, event_type: str) -> str:
    return f"{event_type}: {artifact.name} ({artifact.artifact_type}) — {artifact.full_path}"


def _assign_clusters(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """
    Sort events by timestamp and assign cluster IDs.
    A new cluster starts whenever the gap to the previous event exceeds 5 minutes.
    """
    if not events:
        return events

    events_sorted = sorted(events, key=lambda e: e.timestamp)
    cluster_id = 0
    prev_ts: Optional[datetime] = None
    result: list[TimelineEvent] = []

    for event in events_sorted:
        if prev_ts is None or (event.timestamp - prev_ts) > _CLUSTER_WINDOW:
            cluster_id += 1
        result.append(TimelineEvent(
            artifact_id=event.artifact_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            description=event.description,
            cluster_id=cluster_id,
        ))
        prev_ts = event.timestamp

    return result


def run_timeline(db: DatabaseHandler) -> int:
    """
    Build and persist timeline events.

    Returns
    -------
    int
        Number of events written to timeline_events.
    """
    artifacts = db.get_triaged_artifacts_with_timestamps()
    if not artifacts:
        logger.warning("No artifacts with timestamps found — timeline will be empty.")
        return 0

    logger.info("Building timeline from %d timestamped artifacts.", len(artifacts))

    raw_events: list[TimelineEvent] = []
    for artifact in artifacts:
        for ts, event_type in _events_from_artifact(artifact):
            raw_events.append(TimelineEvent(
                artifact_id=artifact.artifact_id,
                timestamp=ts,
                event_type=event_type,
                description=_build_description(artifact, event_type),
                cluster_id=None,
            ))

    clustered = _assign_clusters(raw_events)
    db.insert_many_timeline_events(clustered)

    cluster_count = max((e.cluster_id or 0) for e in clustered) if clustered else 0
    logger.info(
        "Timeline complete: %d events across %d activity clusters.",
        len(clustered), cluster_count,
    )
    return len(clustered)
