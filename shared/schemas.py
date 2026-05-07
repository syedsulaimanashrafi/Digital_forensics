"""
Pydantic v2 models shared across all pipeline agents.
These mirror the SQLite table schemas exactly so round-trips are lossless.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TriagedArtifact(BaseModel):
    artifact_id: str                  # ART_00001 format — assigned by triage agent
    name: str
    full_path: str
    artifact_type: str                # FILE, REGISTRY, URL, USB, EMAIL, …
    source_module: str                # which Autopsy export CSV it came from
    score: int                        # WFS composite score
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    accessed: Optional[datetime] = None
    md5_hash: Optional[str] = None
    raw_value: Optional[str] = None


class NSRLDiscarded(BaseModel):
    """Artifact removed by NSRL known-good filter."""
    artifact_id: str
    name: str
    full_path: str
    md5_hash: Optional[str] = None
    source_module: str


class BelowThreshold(BaseModel):
    """Artifact that was scored but fell outside Top-K."""
    artifact_id: str
    name: str
    full_path: str
    artifact_type: str
    source_module: str
    score: int
    md5_hash: Optional[str] = None


class TimelineEvent(BaseModel):
    artifact_id: str
    timestamp: datetime
    event_type: str                   # CREATED, MODIFIED, ACCESSED
    description: str
    cluster_id: Optional[int] = None  # 5-minute activity burst id


class IOCEntity(BaseModel):
    artifact_id: str
    entity_type: str                  # IP, EMAIL, HASH, DOMAIN, USERNAME, MAC
    value: str
    classification: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class NarrativeClaim(BaseModel):
    text: str
    citation_tags: List[str]          # extracted [ART_XXXXX] or [INFERENCE] tags
    inference_count: int


class NarrativeOutput(BaseModel):
    run_id: str
    full_text: str
    claim_count: int
    created_at: datetime


class VerifierResult(BaseModel):
    claim_text: str
    status: str                       # GROUNDED | UNCITED | INVENTED | DISTORTED
    artifact_id: Optional[str] = None
    reason: str


class RunLogEntry(BaseModel):
    timestamp: datetime
    run_id: str
    agent: str
    model: str
    prompt_version: str
    temperature: float
    tokens_in: int
    tokens_out: int
    duration_ms: int
    error_state: Optional[str] = None
