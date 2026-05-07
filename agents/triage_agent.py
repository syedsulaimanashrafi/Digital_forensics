"""
Triage Agent — deterministic, no LLM.

Three-pass pipeline:
  Pass 1  NSRL filter   — discard artifacts whose MD5 matches the known-good set
  Pass 2  WFS scoring   — weighted feature scoring matrix
  Pass 3  Top-K select  — sort desc, take TOP_K_TRIAGE, assign ART_XXXXX IDs

All CSVs from AUTOPSY_EXPORT_PATH are read and normalised to TriagedArtifact.
Results land in the triaged_artifacts, nsrl_discarded, and below_threshold tables.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from shared.config import AUTOPSY_EXPORT_PATH, TOP_K_TRIAGE
from shared.database import DatabaseHandler
from shared.nsrl_loader import NSRLLoader
from shared.schemas import BelowThreshold, NSRLDiscarded, TriagedArtifact

logger = logging.getLogger(__name__)

# ── WFS scoring constants ─────────────────────────────────────────────────────

_PERSISTENCE_PATHS = re.compile(
    r"(\\CurrentVersion\\Run|\\CurrentVersion\\RunOnce"
    r"|\\ScheduledTasks|\\Tasks\\Microsoft\\Windows\\)"
    r"|(software\\microsoft\\windows\\currentversion\\run)",
    re.IGNORECASE,
)
_EXECUTION_PATHS = re.compile(
    r"(\\Prefetch\\|\\UserAssist\\|Amcache\.hve|\\AppCompatCache|\\RecentApps"
    r"|run_programs|runprograms)",
    re.IGNORECASE,
)
_INTERESTING_MODULES = re.compile(
    r"(interesting_items?|interesting_files?|notable_items?)",
    re.IGNORECASE,
)
_SUSPICIOUS_SCORE_MODULES = re.compile(
    r"suspicious_items?",
    re.IGNORECASE,
)
_SHELL_BAG_MODULES = re.compile(r"shell.?bag", re.IGNORECASE)
_METADATA_MODULES  = re.compile(r"\bmetadata\b", re.IGNORECASE)
_KEYWORD_MODULES   = re.compile(r"keyword", re.IGNORECASE)
_CLOUD_INDICATORS = re.compile(
    r"(Google Drive|OneDrive|iCloud|Dropbox)", re.IGNORECASE
)
_USB_INDICATORS = re.compile(
    r"(USBSTOR|USB\\|\\Device\\HarddiskVolume|usb_devices)", re.IGNORECASE
)
_SUSPICIOUS_PATHS = re.compile(
    r"(\\Temp\\|\\Downloads\\|\\AppData\\Roaming\\)", re.IGNORECASE
)
_SUSPICIOUS_EXTS = re.compile(
    r"\.(exe|ps1|bat|vbs|dll|hta|scr|com|cmd|jar)$", re.IGNORECASE
)
_USER_ACTIVITY = re.compile(
    r"(web_history|web_bookmark|RecentDocs|recent_documents|web_search)",
    re.IGNORECASE,
)
_USER_PROFILE = re.compile(r"(\\Users\\|\\Desktop\\|\\Documents\\)", re.IGNORECASE)
_EMAIL_INDICATORS = re.compile(r"(\.eml$|\.pst$|@[a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE)


def _wfs_score(path: str, source_module: str, flags: dict) -> int:
    score = 0
    combined = f"{path} {source_module}"

    # ── Autopsy explicit flags (evaluated first, additive) ────────────────────
    # interesting_items: analyst-defined rule matches (~6 items) — guaranteed Top-K
    if flags.get("interesting_item"):
        score += 20
    # suspicious_items: Autopsy automated scoring (~4,770 items) — elevated but stackable
    if flags.get("suspicious_item"):
        score += 8
    if flags.get("encrypted"):
        score += 5
    if flags.get("extension_mismatch"):
        score += 4

    # ── Heuristic signals ─────────────────────────────────────────────────────
    if _PERSISTENCE_PATHS.search(combined):
        score += 5
    if _EXECUTION_PATHS.search(combined):
        score += 4
    if _SHELL_BAG_MODULES.search(source_module):
        score += 4                       # folder navigation history
    if _CLOUD_INDICATORS.search(combined):
        score += 4
    if _USB_INDICATORS.search(combined):
        score += 4
    if _KEYWORD_MODULES.search(source_module):
        score += 2                       # base signal; stacks with path/ext rules
    if _SUSPICIOUS_PATHS.search(path):
        score += 3
    if _SUSPICIOUS_EXTS.search(path):
        score += 3
    if _USER_ACTIVITY.search(source_module):
        score += 3
    if _EMAIL_INDICATORS.search(path):
        score += 3
    if _USER_PROFILE.search(path):
        score += 2
    if _METADATA_MODULES.search(source_module):
        score += 2                       # document author/org metadata

    return score


# ── CSV normalisation ─────────────────────────────────────────────────────────

_TIMESTAMP_COLS = [
    "created", "created_time", "date_created",
    "modified", "modified_time", "date_modified", "last_modified",
    "accessed", "accessed_time", "date_accessed", "last_accessed",
]

_KNOWN_COLUMN_MAPS = {
    # canonical name → possible CSV column names (lowercase)
    "name":         ["name", "file_name", "filename", "object_name", "artifact_name",
                     "source name",                        # Autopsy analysis result exports
                     "program name", "program_name",       # Run Programs
                     "keyword",                            # Keyword Hits
                     "list name",                          # Keyword Hits (list summary)
                     "set name", "set_name", "rule name", "rule_name"],  # Interesting Items
    "full_path":    ["full_path", "path", "file_path", "file path", "location",
                     "object_id", "value", "data",
                     "source file", "source_file",         # Keyword Hits
                     "source path", "source_path"],
    "md5_hash":     ["md5", "md5_hash", "hash_md5", "md5sum"],
    "created":      ["created", "created_time", "date_created", "cr_time",
                     "created date",                       # suspicious_items.csv
                     "date/time"],                         # usb_device_attached, run_programs fallback
    "modified":     ["modified", "modified_time", "date_modified", "m_time",
                     "last run", "last_run", "last run date", "date_last_run",  # Run Programs
                     "last write",                         # Shell Bags
                     "date/time"],                         # run_programs last-run time
    "accessed":     ["accessed", "accessed_time", "date_accessed", "a_time",
                     "date accessed"],
    "raw_value":    ["raw_value", "value", "data", "text_result",
                     "arguments", "run count", "run_count", "count",  # Run Programs
                     "preview", "preview text", "preview_text",        # Keyword Hits
                     "comment", "category", "description", "details",  # Interesting/Suspicious
                     "conclusion", "justification",                    # Encryption/Mismatch
                     "extension", "mime type", "mime_type",            # Extension Mismatch
                     "artifact value", "artifact_value",               # Metadata
                     "attribute value", "attribute_value",
                     "key",                                            # Shell Bags (registry key)
                     "source",                                         # Shell Bags (hive source)
                     "cookie value", "cookie_value",                   # Web Cookies
                     "device make", "device model", "device id",       # USB
                     "type"],                                          # Suspicious Items type field
    # Shell Bags folder path (fallback if 'path' column absent)
    "shell_folder": ["folder path", "folder_path", "folder"],
    # Metadata source file and attribute type
    "meta_source":  ["source file", "source_file", "artifact source", "artifact_source"],
    "meta_type":    ["artifact type", "artifact_type", "attribute type",
                     "attribute_type", "metadata type", "metadata_type"],
}


def _find_col(df_cols_lower: list[str], candidates: list[str]) -> Optional[str]:
    """Return first matching column name (case-insensitive), or None."""
    for c in candidates:
        if c in df_cols_lower:
            return c
    return None


_TZ_SUFFIX = re.compile(r'\s+[A-Z]{2,5}$')   # strip "CEST", "EST", "UTC" etc.

def _parse_datetime(val: object) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "nat", "none", ""):
        return None
    s = _TZ_SUFFIX.sub("", s).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _infer_artifact_type(source_module: str, path: str) -> str:
    sl = source_module.lower()
    if _INTERESTING_MODULES.search(sl):
        return "INTERESTING"
    if _SUSPICIOUS_SCORE_MODULES.search(sl):
        return "SUSPICIOUS"
    if _SHELL_BAG_MODULES.search(sl):
        return "SHELL_BAG"
    if _METADATA_MODULES.search(sl):
        return "METADATA"
    if _KEYWORD_MODULES.search(sl):
        return "KEYWORD_HIT"
    if "usb" in sl:
        return "USB"
    if "registry" in sl or "regedit" in sl or "hive" in sl:
        return "REGISTRY"
    if "web" in sl or "url" in sl or "browser" in sl or "cookie" in sl:
        return "URL"
    if "email" in sl or ".eml" in path.lower() or ".pst" in path.lower():
        return "EMAIL"
    if "prefetch" in sl or "amcache" in sl or "userassist" in sl or "run_program" in sl:
        return "EXECUTION"
    return "FILE"


def _normalise_csv(csv_path: Path) -> list[dict]:
    """
    Read one Autopsy export CSV and return a list of raw row dicts
    normalised to the canonical field names.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False, dtype=str)
    except Exception as exc:
        logger.error("Failed to read %s: %s", csv_path, exc)
        return []

    if df.empty:
        logger.warning("Empty CSV: %s", csv_path.name)
        return []

    # Build a lower-case column lookup
    col_map: dict[str, str] = {c.lower().strip(): c for c in df.columns}
    lower_cols = list(col_map.keys())

    def get(df_row: pd.Series, canonical: str) -> Optional[str]:
        col_key = _find_col(lower_cols, _KNOWN_COLUMN_MAPS.get(canonical, [canonical]))
        if col_key is None:
            return None
        raw = df_row.get(col_map[col_key])
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        return str(raw).strip() or None

    source = csv_path.stem
    is_interesting = bool(_INTERESTING_MODULES.search(source))
    is_suspicious  = bool(_SUSPICIOUS_SCORE_MODULES.search(source))
    is_shell_bag   = bool(_SHELL_BAG_MODULES.search(source))
    is_metadata    = bool(_METADATA_MODULES.search(source))

    rows = []
    for _, row in df.iterrows():
        name = get(row, "name") or csv_path.stem

        # Shell Bags store the accessed folder in a dedicated column
        if is_shell_bag:
            full_path = get(row, "shell_folder") or get(row, "full_path") or name
        # Metadata stores the source file path separately
        elif is_metadata:
            full_path = get(row, "meta_source") or get(row, "full_path") or name
            meta_type = get(row, "meta_type")
            if meta_type:
                name = f"{meta_type}: {name}"
        else:
            full_path = get(row, "full_path") or name

        # Suspicious items: Source column is empty; derive name from filename in Path
        if is_suspicious or is_interesting:
            fname = full_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            if fname and fname != full_path:
                name = fname

        # Autopsy flag columns
        flags = {
            "encrypted":          str(get(row, "encrypted") or "").lower() in ("true", "yes", "1"),
            "extension_mismatch": str(get(row, "extension_mismatch") or "").lower() in ("true", "yes", "1"),
            # encryption_suspected / extension_mismatch CSVs have a Score column
            "interesting_item":   is_interesting,
            "suspicious_item":    is_suspicious,
        }

        # Encryption / mismatch CSVs: treat every row as carrying the flag
        if "encryption_suspected" in source.lower():
            flags["encrypted"] = True
        if "extension_mismatch" in source.lower():
            flags["extension_mismatch"] = True

        # USB: concatenate device fields into raw_value for richer IOC extraction
        if "usb" in source.lower():
            parts = [get(row, k) for k in ("device make", "device model", "device id") if get(row, k)]
            if parts:
                raw_value_override = " | ".join(p for p in parts if p)
            else:
                raw_value_override = None
        else:
            raw_value_override = None

        rows.append({
            "name":          name,
            "full_path":     full_path,
            "source_module": source,
            "md5_hash":      get(row, "md5_hash"),
            "created":       _parse_datetime(get(row, "created")),
            "modified":      _parse_datetime(get(row, "modified")),
            "accessed":      _parse_datetime(get(row, "accessed")),
            "raw_value":     raw_value_override or get(row, "raw_value"),
            "flags":         flags,
        })

    logger.debug("Normalised %d rows from %s", len(rows), csv_path.name)
    return rows


# ── Main agent function ───────────────────────────────────────────────────────

def run_triage(db: DatabaseHandler, nsrl: NSRLLoader) -> int:
    """
    Execute the three-pass triage pipeline.

    Returns
    -------
    int
        Number of artifacts written to triaged_artifacts.
    """
    export_dir = AUTOPSY_EXPORT_PATH
    csv_files = sorted(export_dir.glob("*.csv"))

    if not csv_files:
        logger.warning("No CSV files found in %s", export_dir)
        return 0

    logger.info("Found %d CSV export(s) in %s", len(csv_files), export_dir)

    # ── Pass 0 — load all rows ────────────────────────────────────────────────
    all_rows: list[dict] = []
    for csv_path in csv_files:
        all_rows.extend(_normalise_csv(csv_path))

    logger.info("Total rows after normalisation: %d", len(all_rows))

    # ── Pass 1 — NSRL filter ──────────────────────────────────────────────────
    hashes = {r["md5_hash"] for r in all_rows if r["md5_hash"]}
    nsrl_hits = nsrl.contains_many(hashes)
    logger.info("NSRL filter: %d unique hashes checked, %d matched known-good",
                len(hashes), len(nsrl_hits))

    clean_rows: list[dict] = []
    discarded: list[NSRLDiscarded] = []
    _discard_counter = 0

    for row in all_rows:
        if row["md5_hash"] and row["md5_hash"].upper() in nsrl_hits:
            _discard_counter += 1
            discarded.append(NSRLDiscarded(
                artifact_id=f"DIS_{_discard_counter:05d}",
                name=row["name"],
                full_path=row["full_path"],
                md5_hash=row["md5_hash"],
                source_module=row["source_module"],
            ))
        else:
            clean_rows.append(row)

    if discarded:
        db.insert_many_nsrl_discarded(discarded)
        logger.info("NSRL discarded: %d artifacts", len(discarded))

    # ── Pass 2 — WFS scoring ──────────────────────────────────────────────────
    scored: list[tuple[int, dict]] = []
    for row in clean_rows:
        score = _wfs_score(row["full_path"], row["source_module"], row["flags"])
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    logger.info("Scored %d artifacts (max score=%d, min score=%d)",
                len(scored),
                scored[0][0] if scored else 0,
                scored[-1][0] if scored else 0)

    # ── Pass 3 — Diversity-aware Top-K selection and ID assignment ───────────
    # Cap any single source_module at TOP_K // 4 slots so high-volume modules
    # (suspicious_items: 4,769 rows; web_history: 887 rows) can't crowd out
    # small but forensically critical modules (run_programs, USB, shell_bags).
    # After the cap, remaining slots are filled from the overflow in score order.
    per_module_cap = max(50, TOP_K_TRIAGE // 4)
    module_counts: dict[str, int] = {}
    primary: list[tuple[int, dict]] = []
    overflow: list[tuple[int, dict]] = []

    for item in scored:
        src = item[1]["source_module"]
        if module_counts.get(src, 0) < per_module_cap:
            primary.append(item)
            module_counts[src] = module_counts.get(src, 0) + 1
        else:
            overflow.append(item)

    # Fill remaining slots from overflow (still score-ordered)
    combined = primary + overflow
    top_k   = combined[:TOP_K_TRIAGE]
    below_k = combined[TOP_K_TRIAGE:]

    triaged: list[TriagedArtifact] = []
    for idx, (score, row) in enumerate(top_k, start=1):
        artifact_id = f"ART_{idx:05d}"
        triaged.append(TriagedArtifact(
            artifact_id=artifact_id,
            name=row["name"],
            full_path=row["full_path"],
            artifact_type=_infer_artifact_type(row["source_module"], row["full_path"]),
            source_module=row["source_module"],
            score=score,
            created=row["created"],
            modified=row["modified"],
            accessed=row["accessed"],
            md5_hash=row["md5_hash"],
            raw_value=row["raw_value"],
        ))

    below: list[BelowThreshold] = []
    for idx, (score, row) in enumerate(below_k, start=1):
        below.append(BelowThreshold(
            artifact_id=f"BLW_{idx:05d}",
            name=row["name"],
            full_path=row["full_path"],
            artifact_type=_infer_artifact_type(row["source_module"], row["full_path"]),
            source_module=row["source_module"],
            score=score,
            md5_hash=row["md5_hash"],
        ))

    if triaged:
        db.insert_many_triaged_artifacts(triaged)
    if below:
        db.insert_many_below_threshold(below)

    logger.info(
        "Triage complete: %d artifacts accepted, %d below threshold",
        len(triaged), len(below),
    )
    return len(triaged)
