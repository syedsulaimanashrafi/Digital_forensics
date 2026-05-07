"""
NSRL known-good hash lookup.

On first call, reads the raw NSRLFile.txt (pipe-delimited or CSV) and
builds a single-column SQLite index for O(1) MD5 lookups.
The raw file is never loaded into memory — it is streamed row by row.

NSRL file header format:
  "SHA-1","MD5","CRC32","FileName","FileSize","ProductCode","OpSystemCode","SpecialCode"
MD5 is at column index 1 (0-based).
"""
from __future__ import annotations

import csv
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Set

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100_000


class NSRLLoader:
    """Provides fast MD5 hash lookups against the NSRL dataset."""

    def __init__(self, nsrl_file: Path, index_db: Path) -> None:
        self.nsrl_file = nsrl_file
        self.index_db = index_db

        if not self._index_is_ready():
            if not nsrl_file.exists():
                logger.warning(
                    "NSRL file not found at %s — NSRL filtering disabled. "
                    "Place NSRLFile.txt there to enable it.", nsrl_file
                )
            else:
                logger.info("Building NSRL index at %s (one-time operation)…", index_db)
                self._build_index()
                logger.info("NSRL index ready.")
        else:
            logger.debug("NSRL index found at %s.", index_db)

    # ── Internal ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.index_db, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nsrl_hashes (
                md5 TEXT PRIMARY KEY
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_md5 ON nsrl_hashes(md5)")

    def _index_is_ready(self) -> bool:
        if not self.index_db.exists():
            return False
        try:
            with self._conn() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM nsrl_hashes"
                ).fetchone()[0]
            return count > 0
        except sqlite3.OperationalError:
            return False

    def _build_index(self) -> None:
        self.index_db.parent.mkdir(parents=True, exist_ok=True)
        insert_sql = "INSERT OR IGNORE INTO nsrl_hashes (md5) VALUES (?)"

        with self._conn() as conn:
            self._create_table(conn)
            conn.commit()

        batch: list[tuple[str]] = []
        rows_written = 0

        with open(self.nsrl_file, encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header

            for row in reader:
                if len(row) < 2:
                    continue
                md5 = row[1].strip().strip('"').upper()
                if len(md5) != 32:
                    continue
                batch.append((md5,))

                if len(batch) >= _BATCH_SIZE:
                    with self._conn() as conn:
                        conn.executemany(insert_sql, batch)
                    rows_written += len(batch)
                    batch.clear()
                    logger.debug("NSRL index: %d hashes written…", rows_written)

        if batch:
            with self._conn() as conn:
                conn.executemany(insert_sql, batch)
            rows_written += len(batch)

        logger.info("NSRL index built: %d hashes indexed.", rows_written)

    # ── Public API ────────────────────────────────────────────────────────────

    def contains(self, md5_hash: str) -> bool:
        """Return True if md5_hash is in the NSRL known-good set."""
        if not self.index_db.exists():
            return False
        normalised = md5_hash.strip().upper()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM nsrl_hashes WHERE md5=?", (normalised,)
            ).fetchone()
        return row is not None

    def contains_many(self, md5_hashes: Iterable[str]) -> Set[str]:
        """
        Batch lookup. Returns the subset of supplied hashes that ARE in NSRL.
        More efficient than calling contains() in a loop.
        """
        if not self.index_db.exists():
            return set()

        hashes = [h.strip().upper() for h in md5_hashes if h]
        if not hashes:
            return set()

        found: Set[str] = set()
        # SQLite IN clause limit is ~999 params; process in chunks
        chunk_size = 500
        with self._conn() as conn:
            for i in range(0, len(hashes), chunk_size):
                chunk = hashes[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT md5 FROM nsrl_hashes WHERE md5 IN ({placeholders})",
                    chunk,
                ).fetchall()
                found.update(r[0] for r in rows)
        return found

    def index_count(self) -> int:
        """Return number of hashes in the index (for diagnostics)."""
        if not self.index_db.exists():
            return 0
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM nsrl_hashes").fetchone()[0]
