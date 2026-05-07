"""
Central configuration loader. All settings come from .env — no hardcoded
paths or values anywhere in the pipeline.
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

# Project root is two levels up from shared/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _path(key: str, default: str) -> Path:
    raw = os.getenv(key, default)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ── Runtime settings ──────────────────────────────────────────────────────────

OLLAMA_HOST: str       = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str      = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TEMPERATURE: float = _float("LLM_TEMPERATURE", 0.0)

DATABASE_PATH: Path      = _path("DATABASE_PATH",      "./data/forensics.db")
NSRL_HASH_PATH: Path     = _path("NSRL_HASH_PATH",     "./data/nsrl/NSRLFile.txt")
AUTOPSY_EXPORT_PATH: Path = _path("AUTOPSY_EXPORT_PATH", "./data/cfreds_exports")

TOP_K_TRIAGE: int    = _int("TOP_K_TRIAGE", 400)
PROMPT_VERSION: str  = os.getenv("PROMPT_VERSION", "narrative_v1")
LOG_LEVEL: str       = os.getenv("LOG_LEVEL", "INFO")

# Derived paths
NSRL_INDEX_PATH: Path = NSRL_HASH_PATH.parent / "nsrl_index.db"
PROMPTS_DIR: Path     = PROJECT_ROOT / "prompts"
RUN_LOG_PATH: Path    = PROJECT_ROOT / "run_log.jsonl"
OUTPUT_DIR: Path      = PROJECT_ROOT / "output" / "reports"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
