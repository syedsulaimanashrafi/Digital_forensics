"""Tests for the timeline agent."""
from datetime import datetime, timedelta

import pytest

from agents.timeline_agent import _assign_clusters, _events_from_artifact, run_timeline
from shared.database import DatabaseHandler
from shared.schemas import TimelineEvent, TriagedArtifact


def _make_artifact(**kwargs) -> TriagedArtifact:
    defaults = dict(
        artifact_id="ART_00001",
        name="test.exe",
        full_path=r"C:\Temp\test.exe",
        artifact_type="FILE",
        source_module="file_system",
        score=5,
    )
    return TriagedArtifact(**{**defaults, **kwargs})


@pytest.fixture()
def tmp_db(tmp_path):
    db = DatabaseHandler(tmp_path / "test.db")
    return db


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_events_from_artifact_all_timestamps():
    a = _make_artifact(
        created=datetime(2023, 1, 1, 10, 0),
        modified=datetime(2023, 1, 1, 11, 0),
        accessed=datetime(2023, 1, 1, 12, 0),
    )
    pairs = _events_from_artifact(a)
    types = [t for _, t in pairs]
    assert "CREATED" in types
    assert "MODIFIED" in types
    assert "ACCESSED" in types


def test_events_from_artifact_no_timestamps():
    a = _make_artifact()
    assert _events_from_artifact(a) == []


def test_clustering_two_events_same_window():
    base = datetime(2023, 1, 1, 10, 0)
    events = [
        TimelineEvent(artifact_id="ART_00001", timestamp=base,
                      event_type="CREATED", description="A"),
        TimelineEvent(artifact_id="ART_00002", timestamp=base + timedelta(minutes=2),
                      event_type="MODIFIED", description="B"),
    ]
    clustered = _assign_clusters(events)
    assert clustered[0].cluster_id == clustered[1].cluster_id


def test_clustering_two_events_different_windows():
    base = datetime(2023, 1, 1, 10, 0)
    events = [
        TimelineEvent(artifact_id="ART_00001", timestamp=base,
                      event_type="CREATED", description="A"),
        TimelineEvent(artifact_id="ART_00002", timestamp=base + timedelta(minutes=10),
                      event_type="MODIFIED", description="B"),
    ]
    clustered = _assign_clusters(events)
    assert clustered[0].cluster_id != clustered[1].cluster_id


def test_clustering_empty():
    assert _assign_clusters([]) == []


# ── Integration tests ─────────────────────────────────────────────────────────

def test_timeline_with_minimal_input(tmp_db):
    artifact = _make_artifact(
        created=datetime(2023, 6, 1, 9, 0),
        modified=datetime(2023, 6, 1, 9, 30),
    )
    tmp_db.insert_triaged_artifact(artifact)

    count = run_timeline(tmp_db)
    assert count == 2  # one CREATED + one MODIFIED event

    events = tmp_db.get_all_timeline_events()
    assert len(events) == 2
    assert all(e.cluster_id is not None for e in events)


def test_timeline_with_empty_input(tmp_db):
    count = run_timeline(tmp_db)
    assert count == 0


def test_timeline_with_malformed_input(tmp_db):
    """Artifact with all-null timestamps should produce zero events."""
    artifact = _make_artifact()  # no timestamps
    tmp_db.insert_triaged_artifact(artifact)

    count = run_timeline(tmp_db)
    assert count == 0
