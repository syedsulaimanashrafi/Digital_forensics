"""Tests for the verifier agent."""
import pytest

from agents.verifier_agent import _classify_sentence, _split_sentences, run_verifier
from shared.database import DatabaseHandler
from shared.schemas import TriagedArtifact


def _make_artifact(artifact_id="ART_00001") -> TriagedArtifact:
    return TriagedArtifact(
        artifact_id=artifact_id, name="test.exe",
        full_path=r"C:\Temp\test.exe", artifact_type="FILE",
        source_module="file_system", score=5,
    )


@pytest.fixture()
def tmp_db(tmp_path):
    return DatabaseHandler(tmp_path / "test.db")


@pytest.fixture()
def db_with_artifact(tmp_db):
    tmp_db.insert_triaged_artifact(_make_artifact("ART_00001"))
    return tmp_db


# ── Sentence splitting ────────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "First sentence. Second sentence. Third sentence."
    sentences = _split_sentences(text)
    assert len(sentences) >= 2


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_split_sentences_preserves_headers():
    text = "## Executive Summary\nSome sentence. Another."
    sentences = _split_sentences(text)
    assert any(s.startswith("##") for s in sentences)


# ── Claim classification ──────────────────────────────────────────────────────

def test_classify_grounded(db_with_artifact):
    result = _classify_sentence(
        "The file was executed from C:\\Temp. [ART_00001]",
        db_with_artifact, set(), set(),
    )
    assert result.status == "GROUNDED"
    assert result.artifact_id == "ART_00001"


def test_classify_uncited(db_with_artifact):
    result = _classify_sentence(
        "The file was executed from C:\\Temp.",
        db_with_artifact, set(), set(),
    )
    assert result.status == "UNCITED"


def test_classify_invented(db_with_artifact):
    result = _classify_sentence(
        "A malicious script ran on the system. [ART_99999]",
        db_with_artifact, set(), set(),
    )
    assert result.status == "INVENTED"


def test_classify_inference(db_with_artifact):
    result = _classify_sentence(
        "The attacker likely had prior access. [INFERENCE]",
        db_with_artifact, set(), set(),
    )
    assert result.status == "INFERENCE"


def test_classify_distorted_below_threshold(tmp_db):
    """Citation to a below-threshold artifact should be DISTORTED."""
    tmp_db.insert_triaged_artifact(_make_artifact("ART_00001"))
    below_ids = {"ART_99999"}
    result = _classify_sentence(
        "A suspicious process ran. [ART_99999]",
        tmp_db, set(), below_ids,
    )
    assert result.status == "DISTORTED"


# ── Integration tests ─────────────────────────────────────────────────────────

def test_verifier_with_minimal_input(tmp_db, tmp_path):
    tmp_db.insert_triaged_artifact(_make_artifact("ART_00001"))
    tmp_db.insert_narrative_output(
        run_id="test-run",
        full_text="The file was found in the Temp directory. [ART_00001]",
        claim_count=1,
    )

    import agents.verifier_agent as va
    va.OUTPUT_DIR = tmp_path / "reports"

    summary = run_verifier(tmp_db, "test-run")
    assert summary["grounded"] >= 1
    assert summary["hallucination_rate"] == 0.0


def test_verifier_with_empty_input(tmp_db):
    result = run_verifier(tmp_db, "missing-run")
    assert "error" in result


def test_verifier_with_malformed_input(tmp_db, tmp_path):
    """Narrative with no citations at all should yield 100% hallucination rate."""
    tmp_db.insert_triaged_artifact(_make_artifact("ART_00001"))
    tmp_db.insert_narrative_output(
        run_id="bad-run",
        full_text=(
            "The suspect accessed the system. "
            "Multiple files were downloaded. "
            "Evidence points to data exfiltration."
        ),
        claim_count=0,
    )

    import agents.verifier_agent as va
    va.OUTPUT_DIR = tmp_path / "reports"

    summary = run_verifier(tmp_db, "bad-run")
    assert summary["hallucination_rate"] == 1.0
    assert summary["uncited"] > 0
