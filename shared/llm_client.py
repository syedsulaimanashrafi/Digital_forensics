"""
Ollama wrapper with automatic run-log recording and Instructor integration.

Every call is logged to the SQLite run_log table AND the JSONL audit file.
Temperature is always forced to 0 regardless of caller preference.
On schema-validation failure the call is retried exactly once.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Type, TypeVar

import instructor
import ollama
from pydantic import BaseModel

from shared.config import (
    LLM_TEMPERATURE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    PROMPT_VERSION,
    RUN_LOG_PATH,
)
from shared.database import DatabaseHandler
from shared.schemas import RunLogEntry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        client.list()
        return True
    except Exception as exc:
        logger.error("Ollama not reachable at %s: %s", OLLAMA_HOST, exc)
        return False


class LLMClient:
    """
    Thin wrapper around the Ollama Python SDK.

    Parameters
    ----------
    db : DatabaseHandler
        Used to persist every call to run_log.
    run_id : str
        Identifier for the current pipeline run.
    agent_name : str
        Which agent is making the call (recorded in run_log).
    model : str
        Ollama model tag; defaults to OLLAMA_MODEL from config.
    """

    def __init__(
        self,
        db: DatabaseHandler,
        run_id: str,
        agent_name: str,
        model: str = OLLAMA_MODEL,
    ) -> None:
        self.db = db
        self.run_id = run_id
        self.agent_name = agent_name
        self.model = model
        self._raw_client = ollama.Client(host=OLLAMA_HOST)
        self._instructor_client = instructor.from_ollama(
            self._raw_client,
            mode=instructor.Mode.JSON,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Plain text generation — no schema enforcement.
        Returns the assistant message content as a string.
        """
        messages = self._build_messages(prompt, system)
        start = time.monotonic()
        error_state: Optional[str] = None
        response_text = ""
        tokens_in = tokens_out = 0

        try:
            response = self._raw_client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": LLM_TEMPERATURE},
            )
            response_text = response.message.content or ""
            tokens_in  = response.prompt_eval_count or 0
            tokens_out = response.eval_count or 0
        except Exception as exc:
            error_state = str(exc)
            logger.error("[%s] LLM chat failed: %s", self.agent_name, exc)
            raise
        finally:
            self._log(
                prompt=prompt,
                duration_ms=int((time.monotonic() - start) * 1000),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error_state=error_state,
            )

        return response_text

    def chat_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        max_retries: int = 1,
    ) -> T:
        """
        Structured generation with Instructor schema enforcement.
        Retries once on validation failure.
        """
        messages = self._build_messages(prompt, system)
        start = time.monotonic()
        error_state: Optional[str] = None
        tokens_in = tokens_out = 0

        try:
            result: T = self._instructor_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
                options={"temperature": LLM_TEMPERATURE},
            )
            # Instructor does not always expose token counts; guard gracefully
            tokens_in  = getattr(result, "_raw_response", {}).get("prompt_eval_count", 0) or 0
            tokens_out = getattr(result, "_raw_response", {}).get("eval_count", 0) or 0
        except Exception as exc:
            error_state = str(exc)
            logger.error(
                "[%s] Structured LLM call failed (%s): %s",
                self.agent_name, response_model.__name__, exc,
            )
            raise
        finally:
            self._log(
                prompt=prompt,
                duration_ms=int((time.monotonic() - start) * 1000),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error_state=error_state,
            )

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        prompt: str, system: Optional[str]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _log(
        self,
        prompt: str,
        duration_ms: int,
        tokens_in: int,
        tokens_out: int,
        error_state: Optional[str],
    ) -> None:
        entry = RunLogEntry(
            timestamp=datetime.now(timezone.utc),
            run_id=self.run_id,
            agent=self.agent_name,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            temperature=LLM_TEMPERATURE,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            error_state=error_state,
        )
        try:
            self.db.insert_run_log(entry, RUN_LOG_PATH)
        except Exception as log_exc:
            # Logging must never crash the pipeline
            logger.warning("Failed to write run_log entry: %s", log_exc)
