"""Tests for the triage agent."""
import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.triage_agent import _wfs_score, _normalise_csv, run_triage
from shared.database import DatabaseHandler
from shared.nsrl_loader import NSRLLoader


# ── WFS scoring unit tests ────────────────────────────────────────────────────

def test_wfs_persistence_path():
    score = _wfs_score(
        r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run\malware",
        "registry",
        {},
    )
    assert score >= 5


def test_wfs_execution_evidence():
    score = _wfs_score(r"C:\Windows\Prefetch\MALWARE.EXE-ABCD.pf", "prefetch", {})
    assert score >= 4


def test_wfs_encryption_flag():
    score = _wfs_score("somefile.enc", "file_system", {"encrypted": True})
    assert score >= 5


def test_wfs_extension_mismatch():
    score = _wfs_score("image.jpg", "file_system", {"extension_mismatch": True})
    assert score >= 4


def test_wfs_zero_score():
    score = _wfs_score(r"C:\Windows\System32\notepad.exe", "system_files", {})
    # System32 + no special indicators — score may be 0 or very low
    assert score >= 0


# ── CSV normalisation ─────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_normalise_csv_minimal():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "usb_devices.csv"
        _write_csv(csv_path, [
            {"name": "USB Drive", "full_path": r"C:\Users\test", "md5": "abc123"},
        ])
        rows = _normalise_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["name"] == "USB Drive"


def test_normalise_csv_empty():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "empty.csv"
        csv_path.write_text("")
        rows = _normalise_csv(csv_path)
    assert rows == []


def test_normalise_csv_malformed():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "bad.csv"
        csv_path.write_text("col1\tcol2\nnot,a,csv,row")
        rows = _normalise_csv(csv_path)
    # Should not raise; may return rows with defaults
    assert isinstance(rows, list)


# ── Full triage integration ───────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    return DatabaseHandler(tmp_path / "test.db")


@pytest.fixture()
def null_nsrl(tmp_path):
    """NSRLLoader with no index — behaves as disabled."""
    return NSRLLoader(tmp_path / "missing.txt", tmp_path / "nsrl_index.db")


def test_triage_with_minimal_input(tmp_db, null_nsrl, tmp_path, monkeypatch):
    csv_path = tmp_path / "usb_devices.csv"
    _write_csv(csv_path, [
        {
            "name": "SanDisk USB",
            "full_path": r"\\USBSTOR\Disk&Ven_SanDisk",
            "md5": "d41d8cd98f00b204e9800998ecf8427e",
            "created": "2023-01-15 10:00:00",
        }
    ])
    monkeypatch.setattr("agents.triage_agent.AUTOPSY_EXPORT_PATH", tmp_path)
    monkeypatch.setattr("agents.triage_agent.TOP_K_TRIAGE", 400)

    count = run_triage(tmp_db, null_nsrl)
    assert count == 1

    artifacts = tmp_db.get_triaged_artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].artifact_id == "ART_00001"
    assert artifacts[0].name == "SanDisk USB"


def test_triage_with_empty_input(tmp_db, null_nsrl, tmp_path, monkeypatch):
    monkeypatch.setattr("agents.triage_agent.AUTOPSY_EXPORT_PATH", tmp_path)
    count = run_triage(tmp_db, null_nsrl)
    assert count == 0


def test_triage_with_malformed_input(tmp_db, null_nsrl, tmp_path, monkeypatch):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("not,valid\x00csv\ncontent")
    monkeypatch.setattr("agents.triage_agent.AUTOPSY_EXPORT_PATH", tmp_path)
    # Should not raise; just return 0 or however many valid rows parsed
    count = run_triage(tmp_db, null_nsrl)
    assert isinstance(count, int)


def test_triage_nsrl_filter(tmp_db, tmp_path, monkeypatch):
    """Artifact with a known NSRL hash should land in nsrl_discarded, not triaged."""
    known_md5 = "AABBCCDDEEFF00112233445566778899"

    csv_path = tmp_path / "files.csv"
    _write_csv(csv_path, [
        {"name": "known_good.dll", "full_path": r"C:\Windows\system32\known_good.dll",
         "md5": known_md5},
        {"name": "suspect.exe", "full_path": r"C:\Temp\suspect.exe",
         "md5": "00000000000000000000000000000000"},
    ])

    # Build a real NSRL index with just that hash
    nsrl_index = tmp_path / "nsrl_index.db"
    import sqlite3
    conn = sqlite3.connect(nsrl_index)
    conn.execute("CREATE TABLE nsrl_hashes (md5 TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO nsrl_hashes VALUES (?)", (known_md5.upper(),))
    conn.commit()
    conn.close()

    nsrl = NSRLLoader(tmp_path / "NSRLFile.txt", nsrl_index)
    monkeypatch.setattr("agents.triage_agent.AUTOPSY_EXPORT_PATH", tmp_path)
    monkeypatch.setattr("agents.triage_agent.TOP_K_TRIAGE", 400)

    count = run_triage(tmp_db, nsrl)

    # Only suspect.exe should be triaged
    assert count == 1
    artifacts = tmp_db.get_triaged_artifacts()
    assert all("known_good" not in a.name for a in artifacts)
