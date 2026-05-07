"""
SQLite handler — single source of truth between all pipeline agents.
All writes go through this module; no agent touches sqlite3 directly.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from shared.schemas import (
    BelowThreshold,
    IOCEntity,
    NarrativeOutput,
    NSRLDiscarded,
    RunLogEntry,
    TimelineEvent,
    TriagedArtifact,
    VerifierResult,
)

logger = logging.getLogger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS triaged_artifacts (
    artifact_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    full_path     TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    source_module TEXT NOT NULL,
    score         INTEGER NOT NULL,
    created       TEXT,
    modified      TEXT,
    accessed      TEXT,
    md5_hash      TEXT,
    raw_value     TEXT
);

CREATE TABLE IF NOT EXISTS nsrl_discarded (
    artifact_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    full_path     TEXT NOT NULL,
    md5_hash      TEXT,
    source_module TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS below_threshold (
    artifact_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    full_path     TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    source_module TEXT NOT NULL,
    score         INTEGER NOT NULL,
    md5_hash      TEXT
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    description TEXT NOT NULL,
    cluster_id  INTEGER,
    FOREIGN KEY (artifact_id) REFERENCES triaged_artifacts(artifact_id)
);

CREATE TABLE IF NOT EXISTS ioc_entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id    TEXT NOT NULL,
    entity_type    TEXT NOT NULL,
    value          TEXT NOT NULL,
    classification TEXT,
    confidence     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS narrative_output (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    full_text   TEXT NOT NULL,
    claim_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verifier_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    claim_text  TEXT NOT NULL,
    status      TEXT NOT NULL,
    artifact_id TEXT,
    reason      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    agent          TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT,
    temperature    REAL,
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    duration_ms    INTEGER,
    error_state    TEXT
);
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


def _row_to_triaged(row: sqlite3.Row) -> TriagedArtifact:
    return TriagedArtifact(
        artifact_id=row["artifact_id"],
        name=row["name"],
        full_path=row["full_path"],
        artifact_type=row["artifact_type"],
        source_module=row["source_module"],
        score=row["score"],
        created=_str_to_dt(row["created"]),
        modified=_str_to_dt(row["modified"]),
        accessed=_str_to_dt(row["accessed"]),
        md5_hash=row["md5_hash"],
        raw_value=row["raw_value"],
    )


def _row_to_timeline(row: sqlite3.Row) -> TimelineEvent:
    return TimelineEvent(
        artifact_id=row["artifact_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        event_type=row["event_type"],
        description=row["description"],
        cluster_id=row["cluster_id"],
    )


def _row_to_ioc(row: sqlite3.Row) -> IOCEntity:
    return IOCEntity(
        artifact_id=row["artifact_id"],
        entity_type=row["entity_type"],
        value=row["value"],
        classification=row["classification"],
        confidence=row["confidence"],
    )


def _row_to_verifier(row: sqlite3.Row) -> VerifierResult:
    return VerifierResult(
        claim_text=row["claim_text"],
        status=row["status"],
        artifact_id=row["artifact_id"],
        reason=row["reason"],
    )


# ── Handler ───────────────────────────────────────────────────────────────────

class DatabaseHandler:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.debug("DatabaseHandler ready at %s", db_path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_SQL)

    # ── Triage ────────────────────────────────────────────────────────────────

    def insert_triaged_artifact(self, artifact: TriagedArtifact) -> None:
        sql = """
            INSERT OR IGNORE INTO triaged_artifacts
            (artifact_id, name, full_path, artifact_type, source_module,
             score, created, modified, accessed, md5_hash, raw_value)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                artifact.artifact_id, artifact.name, artifact.full_path,
                artifact.artifact_type, artifact.source_module, artifact.score,
                _dt_to_str(artifact.created), _dt_to_str(artifact.modified),
                _dt_to_str(artifact.accessed), artifact.md5_hash, artifact.raw_value,
            ))

    def insert_many_triaged_artifacts(self, artifacts: List[TriagedArtifact]) -> None:
        sql = """
            INSERT OR IGNORE INTO triaged_artifacts
            (artifact_id, name, full_path, artifact_type, source_module,
             score, created, modified, accessed, md5_hash, raw_value)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """
        rows = [
            (a.artifact_id, a.name, a.full_path, a.artifact_type, a.source_module,
             a.score, _dt_to_str(a.created), _dt_to_str(a.modified),
             _dt_to_str(a.accessed), a.md5_hash, a.raw_value)
            for a in artifacts
        ]
        with self._conn() as conn:
            conn.executemany(sql, rows)
        logger.debug("Inserted %d triaged artifacts", len(rows))

    def get_triaged_artifacts(self) -> List[TriagedArtifact]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM triaged_artifacts ORDER BY score DESC"
            ).fetchall()
        return [_row_to_triaged(r) for r in rows]

    def get_triaged_artifacts_with_timestamps(self) -> List[TriagedArtifact]:
        sql = """
            SELECT * FROM triaged_artifacts
            WHERE created IS NOT NULL
               OR modified IS NOT NULL
               OR accessed IS NOT NULL
            ORDER BY COALESCE(created, modified, accessed)
        """
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [_row_to_triaged(r) for r in rows]

    def artifact_exists(self, artifact_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM triaged_artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        return row is not None

    def get_artifact_by_id(self, artifact_id: str) -> Optional[TriagedArtifact]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM triaged_artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        return _row_to_triaged(row) if row else None

    def count_triaged_artifacts(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM triaged_artifacts").fetchone()[0]

    # ── NSRL discarded ────────────────────────────────────────────────────────

    def insert_nsrl_discarded(self, artifact: NSRLDiscarded) -> None:
        sql = """
            INSERT OR IGNORE INTO nsrl_discarded
            (artifact_id, name, full_path, md5_hash, source_module)
            VALUES (?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                artifact.artifact_id, artifact.name, artifact.full_path,
                artifact.md5_hash, artifact.source_module,
            ))

    def insert_many_nsrl_discarded(self, artifacts: List[NSRLDiscarded]) -> None:
        sql = """
            INSERT OR IGNORE INTO nsrl_discarded
            (artifact_id, name, full_path, md5_hash, source_module)
            VALUES (?,?,?,?,?)
        """
        rows = [(a.artifact_id, a.name, a.full_path, a.md5_hash, a.source_module)
                for a in artifacts]
        with self._conn() as conn:
            conn.executemany(sql, rows)

    # ── Below threshold ───────────────────────────────────────────────────────

    def insert_below_threshold(self, artifact: BelowThreshold) -> None:
        sql = """
            INSERT OR IGNORE INTO below_threshold
            (artifact_id, name, full_path, artifact_type, source_module, score, md5_hash)
            VALUES (?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                artifact.artifact_id, artifact.name, artifact.full_path,
                artifact.artifact_type, artifact.source_module, artifact.score,
                artifact.md5_hash,
            ))

    def insert_many_below_threshold(self, artifacts: List[BelowThreshold]) -> None:
        sql = """
            INSERT OR IGNORE INTO below_threshold
            (artifact_id, name, full_path, artifact_type, source_module, score, md5_hash)
            VALUES (?,?,?,?,?,?,?)
        """
        rows = [(a.artifact_id, a.name, a.full_path, a.artifact_type,
                 a.source_module, a.score, a.md5_hash) for a in artifacts]
        with self._conn() as conn:
            conn.executemany(sql, rows)

    def get_below_threshold_artifacts(self) -> List[BelowThreshold]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM below_threshold ORDER BY score DESC"
            ).fetchall()
        return [
            BelowThreshold(
                artifact_id=r["artifact_id"], name=r["name"],
                full_path=r["full_path"], artifact_type=r["artifact_type"],
                source_module=r["source_module"], score=r["score"],
                md5_hash=r["md5_hash"],
            )
            for r in rows
        ]

    # ── Timeline ──────────────────────────────────────────────────────────────

    def insert_timeline_event(self, event: TimelineEvent) -> None:
        sql = """
            INSERT INTO timeline_events
            (artifact_id, timestamp, event_type, description, cluster_id)
            VALUES (?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                event.artifact_id, event.timestamp.isoformat(),
                event.event_type, event.description, event.cluster_id,
            ))

    def insert_many_timeline_events(self, events: List[TimelineEvent]) -> None:
        sql = """
            INSERT INTO timeline_events
            (artifact_id, timestamp, event_type, description, cluster_id)
            VALUES (?,?,?,?,?)
        """
        rows = [(e.artifact_id, e.timestamp.isoformat(), e.event_type,
                 e.description, e.cluster_id) for e in events]
        with self._conn() as conn:
            conn.executemany(sql, rows)
        logger.debug("Inserted %d timeline events", len(rows))

    def get_timeline_events(self, limit: int = 30) -> List[TimelineEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM timeline_events ORDER BY timestamp LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_timeline(r) for r in rows]

    def get_all_timeline_events(self) -> List[TimelineEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM timeline_events ORDER BY timestamp"
            ).fetchall()
        return [_row_to_timeline(r) for r in rows]

    def count_timeline_events(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0]

    # ── IOC entities ──────────────────────────────────────────────────────────

    def insert_ioc_entity(self, entity: IOCEntity) -> None:
        sql = """
            INSERT INTO ioc_entities
            (artifact_id, entity_type, value, classification, confidence)
            VALUES (?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                entity.artifact_id, entity.entity_type, entity.value,
                entity.classification, entity.confidence,
            ))

    def insert_many_ioc_entities(self, entities: List[IOCEntity]) -> None:
        sql = """
            INSERT INTO ioc_entities
            (artifact_id, entity_type, value, classification, confidence)
            VALUES (?,?,?,?,?)
        """
        rows = [(e.artifact_id, e.entity_type, e.value,
                 e.classification, e.confidence) for e in entities]
        with self._conn() as conn:
            conn.executemany(sql, rows)
        logger.debug("Inserted %d IOC entities", len(rows))

    def get_ioc_entities(self) -> List[IOCEntity]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM ioc_entities").fetchall()
        return [_row_to_ioc(r) for r in rows]

    # ── Narrative output ──────────────────────────────────────────────────────

    def insert_narrative_output(self, run_id: str, full_text: str,
                                 claim_count: int) -> None:
        sql = """
            INSERT INTO narrative_output (run_id, full_text, claim_count, created_at)
            VALUES (?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (run_id, full_text, claim_count,
                               datetime.now(timezone.utc).isoformat()))

    def get_narrative_output(self, run_id: str) -> Optional[NarrativeOutput]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM narrative_output WHERE run_id=? ORDER BY id DESC LIMIT 1",
                (run_id,)
            ).fetchone()
        if row is None:
            return None
        return NarrativeOutput(
            run_id=row["run_id"],
            full_text=row["full_text"],
            claim_count=row["claim_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ── Verifier results ──────────────────────────────────────────────────────

    def insert_verifier_result(self, result: VerifierResult, run_id: str) -> None:
        sql = """
            INSERT INTO verifier_results
            (run_id, claim_text, status, artifact_id, reason)
            VALUES (?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                run_id, result.claim_text, result.status,
                result.artifact_id, result.reason,
            ))

    def insert_many_verifier_results(self, results: List[VerifierResult],
                                      run_id: str) -> None:
        sql = """
            INSERT INTO verifier_results
            (run_id, claim_text, status, artifact_id, reason)
            VALUES (?,?,?,?,?)
        """
        rows = [(run_id, r.claim_text, r.status, r.artifact_id, r.reason)
                for r in results]
        with self._conn() as conn:
            conn.executemany(sql, rows)

    def get_verifier_results(self, run_id: str) -> List[VerifierResult]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM verifier_results WHERE run_id=?", (run_id,)
            ).fetchall()
        return [_row_to_verifier(r) for r in rows]

    def get_hallucination_rate(self, run_id: str) -> float:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM verifier_results WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            bad = conn.execute(
                """SELECT COUNT(*) FROM verifier_results
                   WHERE run_id=? AND status IN ('UNCITED','INVENTED','DISTORTED')""",
                (run_id,)
            ).fetchone()[0]
        return bad / total if total > 0 else 0.0

    # ── Run log ───────────────────────────────────────────────────────────────

    def insert_run_log(self, entry: RunLogEntry, jsonl_path: Path) -> None:
        sql = """
            INSERT INTO run_log
            (timestamp, run_id, agent, model, prompt_version, temperature,
             tokens_in, tokens_out, duration_ms, error_state)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                entry.timestamp.isoformat(), entry.run_id, entry.agent,
                entry.model, entry.prompt_version, entry.temperature,
                entry.tokens_in, entry.tokens_out, entry.duration_ms,
                entry.error_state,
            ))
        # Mirror to immutable JSONL audit log
        record = entry.model_dump()
        record["timestamp"] = record["timestamp"].isoformat()
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def get_run_log(self, run_id: Optional[str] = None) -> List[dict]:
        with self._conn() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM run_log WHERE run_id=? ORDER BY id", (run_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM run_log ORDER BY id"
                ).fetchall()
        return [dict(r) for r in rows]
