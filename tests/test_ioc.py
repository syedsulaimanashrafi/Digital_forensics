"""Tests for the IOC agent."""
import pytest

from agents.ioc_agent import _extract_from_text, run_ioc
from shared.database import DatabaseHandler
from shared.schemas import TriagedArtifact


def _make_artifact(artifact_id="ART_00001", name="test", full_path="", raw_value=None,
                   source_module="file_system") -> TriagedArtifact:
    return TriagedArtifact(
        artifact_id=artifact_id, name=name, full_path=full_path,
        artifact_type="FILE", source_module=source_module,
        score=3, raw_value=raw_value,
    )


@pytest.fixture()
def tmp_db(tmp_path):
    return DatabaseHandler(tmp_path / "test.db")


# ── Regex extraction unit tests ───────────────────────────────────────────────

def test_extract_ipv4():
    entities = _extract_from_text("Connected to 192.168.1.100 from host", "ART_00001")
    ips = [e for e in entities if e.entity_type == "IP"]
    assert any(e.value == "192.168.1.100" for e in ips)


def test_extract_email():
    entities = _extract_from_text("user sent mail to attacker@evil.com", "ART_00001")
    emails = [e for e in entities if e.entity_type == "EMAIL"]
    assert any(e.value == "attacker@evil.com" for e in emails)


def test_extract_md5():
    entities = _extract_from_text(
        "hash: d41d8cd98f00b204e9800998ecf8427e", "ART_00001"
    )
    hashes = [e for e in entities if e.entity_type == "HASH" and e.classification == "MD5"]
    assert len(hashes) >= 1


def test_extract_domain():
    entities = _extract_from_text("visited malware.evil.com yesterday", "ART_00001")
    domains = [e for e in entities if e.entity_type == "DOMAIN"]
    assert any("evil.com" in e.value for e in domains)


def test_extract_username_from_path():
    entities = _extract_from_text(
        r"C:\Users\johndoe\Desktop\payload.exe", "ART_00001"
    )
    users = [e for e in entities if e.entity_type == "USERNAME"]
    assert any(e.value == "johndoe" for e in users)


def test_extract_loopback_excluded():
    entities = _extract_from_text("localhost is 127.0.0.1", "ART_00001")
    ips = [e for e in entities if e.entity_type == "IP"]
    assert not any(e.value == "127.0.0.1" for e in ips)


def test_extract_empty_text():
    entities = _extract_from_text("", "ART_00001")
    assert entities == []


def test_extract_no_iocs():
    entities = _extract_from_text("no indicators here just plain text words", "ART_00001")
    assert entities == []


# ── Integration tests ─────────────────────────────────────────────────────────

def test_ioc_with_minimal_input(tmp_db):
    artifact = _make_artifact(
        full_path=r"C:\Users\admin\Desktop\malware.exe",
        raw_value="C2 server: 10.0.0.55",
    )
    tmp_db.insert_triaged_artifact(artifact)

    count = run_ioc(tmp_db, llm=None)
    assert count > 0

    entities = tmp_db.get_ioc_entities()
    types = {e.entity_type for e in entities}
    assert "IP" in types or "USERNAME" in types


def test_ioc_with_empty_input(tmp_db):
    count = run_ioc(tmp_db, llm=None)
    assert count == 0


def test_ioc_with_malformed_input(tmp_db):
    """Artifact with only whitespace/null values should not crash."""
    artifact = _make_artifact(name="   ", full_path="   ", raw_value=None)
    tmp_db.insert_triaged_artifact(artifact)
    count = run_ioc(tmp_db, llm=None)
    assert isinstance(count, int)
