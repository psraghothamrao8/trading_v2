"""LLM classification — §1.

Single entry point: :func:`classify`. Forces strict JSON, validates against a
schema, retries three times with backoff, and caches by content hash in SQLite
so no filing or transcript is ever classified twice.

That cache is not an optimisation. §6.1 polls announcements every 30 seconds;
without deduplication the same filing would be re-classified 120 times an hour,
at cost, with a different answer each time -- and "did we already trade this?"
would have no stable answer.

Prompts live in ``config/prompts/*.txt`` (§9.6), each with a strict JSON schema
and two few-shot examples, so the strategy owner can tune them without touching
Python.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core import clock
from core.config import REPO_ROOT, ConfigError, data_path, get_secrets, get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Classification failed after all retries, or the response was unusable."""


class SchemaError(LLMError):
    """The model returned JSON that does not satisfy the task schema."""


class LLMSetupError(LLMError):
    """The provider is misconfigured: no server, no model, no key.

    Distinct from :class:`SchemaError` because it is **permanent**. Retrying a
    missing model three times just buries the one line telling you to pull it,
    and on CPU inference each pointless retry costs real seconds.
    """


@dataclass
class Classification:
    """A validated classification plus the metadata the journal needs."""

    data: dict[str, Any]
    task: str
    from_cache: bool
    latency_sec: float
    model: str
    attempts: int = 1

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


# ---------------------------------------------------------------------------
# Schema validation
#
# A dependency-free subset of JSON Schema: enough for the three task schemas
# this system uses, and small enough to read in one sitting. Bringing in
# jsonschema for `type`, `enum` and `required` would be more code to audit,
# not less.
# ---------------------------------------------------------------------------


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate(payload: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate ``payload`` against ``schema``. Raises :class:`SchemaError`."""
    types = schema.get("type")
    if types is not None:
        allowed = [types] if isinstance(types, str) else list(types)
        if "null" in allowed and payload is None:
            return
        python_types: tuple[type, ...] = tuple(
            t for name in allowed for t in _TYPE_MAP.get(name, ())
        )
        if python_types:
            # bool is a subclass of int; a boolean is not a number here.
            if isinstance(payload, bool) and "boolean" not in allowed:
                raise SchemaError(f"{path}: expected {allowed}, got boolean")
            if not isinstance(payload, python_types):
                raise SchemaError(
                    f"{path}: expected {allowed}, got {type(payload).__name__}"
                )

    if "enum" in schema and payload not in schema["enum"]:
        raise SchemaError(f"{path}: {payload!r} is not one of {schema['enum']}")

    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        if "minimum" in schema and payload < schema["minimum"]:
            raise SchemaError(f"{path}: {payload} < minimum {schema['minimum']}")
        if "maximum" in schema and payload > schema["maximum"]:
            raise SchemaError(f"{path}: {payload} > maximum {schema['maximum']}")

    if isinstance(payload, dict):
        for key in schema.get("required", []):
            if key not in payload:
                raise SchemaError(f"{path}: missing required key {key!r}")
        for key, sub_schema in (schema.get("properties") or {}).items():
            if key in payload:
                validate(payload[key], sub_schema, f"{path}.{key}")

    if isinstance(payload, list):
        if "maxItems" in schema and len(payload) > schema["maxItems"]:
            raise SchemaError(f"{path}: {len(payload)} items > maxItems {schema['maxItems']}")
        if "minItems" in schema and len(payload) < schema["minItems"]:
            raise SchemaError(f"{path}: {len(payload)} items < minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(payload):
                validate(item, item_schema, f"{path}[{index}]")


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences even when told not to. Rather than
    failing the whole classification on a stray ```json fence, find the
    outermost balanced object.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise SchemaError(f"No JSON object in response: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"Malformed JSON object: {exc}") from exc
    raise SchemaError(f"Unbalanced JSON in response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Prompt loading (§9.6)
# ---------------------------------------------------------------------------


def prompt_dir() -> Path:
    return REPO_ROOT / str(get_settings().get("llm.prompt_dir", "config/prompts"))


def load_prompt(task: str) -> str:
    """Load ``config/prompts/<task>.txt``. Owner-editable (§9.6)."""
    path = prompt_dir() / f"{task}.txt"
    if not path.exists():
        raise ConfigError(
            f"No prompt for task {task!r} at {path}. Prompts are owner-editable "
            f"text files (§9.6); create it with a strict JSON schema and two "
            f"few-shot examples."
        )
    return path.read_text(encoding="utf-8")


def load_schema(task: str) -> dict[str, Any]:
    """Load ``config/prompts/<task>.schema.json``."""
    path = prompt_dir() / f"{task}.schema.json"
    if not path.exists():
        raise ConfigError(f"No JSON schema for task {task!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash TEXT PRIMARY KEY,
    task         TEXT NOT NULL,
    model        TEXT NOT NULL,
    response     TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    hits         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_task ON llm_cache(task, created_at);
"""


class LLMCache:
    """SQLite cache keyed by (task, model, payload) hash (§1)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            configured = str(get_settings().get("llm_cache_db", "") or
                             get_settings().get("storage.llm_cache_db", "data/llm_cache.sqlite"))
            db_path = data_path(Path(configured).name)
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_CACHE_SCHEMA)
            self._conn.commit()

    @staticmethod
    def key(task: str, model: str, payload: dict[str, Any]) -> str:
        """Stable hash over the task, model and payload content."""
        blob = json.dumps(
            {"task": task, "model": model, "payload": payload},
            sort_keys=True, default=str, separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, content_hash: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE content_hash=?", (content_hash,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE llm_cache SET hits = hits + 1 WHERE content_hash=?", (content_hash,)
            )
            self._conn.commit()
        return json.loads(row["response"])

    def put(self, content_hash: str, task: str, model: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(content_hash, task, model, response, created_at, hits) VALUES (?,?,?,?,?,0)",
                (
                    content_hash, task, model,
                    json.dumps(response, default=str),
                    clock.isoformat(clock.now_ist()),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


class LLMClient:
    """Anthropic wrapper with retries, schema validation and caching.

    ``transport`` is injectable: production uses the ``anthropic`` SDK, tests
    pass a callable ``(system, user, max_tokens) -> str``.
    """

    def __init__(
        self,
        transport: Callable[[str, str, int], str] | None = None,
        cache: LLMCache | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        settings = get_settings()
        self.config = settings.section("llm")
        self.provider = str(self.config.get("provider", "anthropic")).lower()
        self.model = str(self.config.require("model"))
        self.max_tokens = int(self.config.get("max_tokens", 1500))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.retries = int(self.config.get("retries", 3))
        self.backoff_base = float(self.config.get("backoff_base_seconds", 2.0))
        self.use_cache = bool(self.config.get("cache_in_sqlite", True))
        self._transport = transport
        self._cache = cache
        self._sleep = sleeper or time.sleep
        self._client: Any = None

        if self.provider not in ("anthropic", "ollama"):
            raise ConfigError(
                f"Unknown llm.provider {self.provider!r}; expected 'anthropic' or 'ollama'"
            )

    @property
    def cache(self) -> Optional[LLMCache]:
        if not self.use_cache:
            return None
        if self._cache is None:
            self._cache = LLMCache()
        return self._cache

    # -- providers --------------------------------------------------------

    def _call_model(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> str:
        """Dispatch to the configured provider. Returns the raw response text."""
        if self._transport is not None:
            return self._transport(system, user, self.max_tokens)
        if self.provider == "ollama":
            return self._call_ollama(system, user, schema)
        return self._call_anthropic(system, user)

    def _call_anthropic(self, system: str, user: str) -> str:
        secrets = get_secrets()
        if not secrets.anthropic_api_key:
            raise LLMSetupError(
                "ANTHROPIC_API_KEY is not set in .env (§0.4). Either add it, or "
                "set `llm.provider: ollama` in config/settings.yaml to run locally."
            )
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise LLMError("anthropic is not installed; pip install -r requirements.txt") from exc
            self._client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    def _call_ollama(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> str:
        """Local inference via the Ollama HTTP API. No API key, no network egress.

        Two things make this workable on a CPU-only machine:

        * ``format`` is set to the task's JSON schema. Ollama constrains
          decoding to that grammar, so a 3B model returns valid JSON instead of
          prose-wrapped near-JSON. Without it the retry loop burns three slow
          CPU generations on formatting failures.
        * ``keep_alive`` holds the model in RAM between calls, and the system
          prompt is byte-identical every time so llama.cpp reuses its KV cache
          for the prefix. The ~2.4k-token prompt is therefore prefilled once
          per model load, not once per filing -- the difference between ~60s
          and ~10s per classification on four cores.
        """
        import httpx

        base_url = str(self.config.get("ollama.base_url", "http://localhost:11434"))
        timeout = float(self.config.get("ollama.timeout_seconds", 300))

        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": int(self.config.get("ollama.num_ctx", 8192)),
        }
        threads = self.config.get("ollama.num_thread", None)
        if threads:
            options["num_thread"] = int(threads)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": str(self.config.get("ollama.keep_alive", "30m")),
            "options": options,
        }
        if schema is not None and self.config.get("ollama.structured_output", True):
            body["format"] = schema

        try:
            response = httpx.post(f"{base_url}/api/chat", json=body, timeout=timeout)
        except Exception as exc:
            raise LLMSetupError(
                f"Cannot reach Ollama at {base_url}: {exc}. "
                f"Is `ollama serve` running? Start it, or pull the model with "
                f"`ollama pull {self.model}`."
            ) from exc

        if response.status_code == 404:
            raise LLMSetupError(
                f"Ollama does not have model {self.model!r}. Run: ollama pull {self.model}"
            )
        if response.status_code >= 400:
            raise LLMError(f"Ollama returned {response.status_code}: {response.text[:300]}")

        payload = response.json()
        content = (payload.get("message") or {}).get("content", "")
        if not content:
            raise LLMError(f"Ollama returned an empty message: {str(payload)[:300]}")
        return content

    def classify(
        self, task: str, payload: dict[str, Any], schema: dict[str, Any] | None = None
    ) -> Classification:
        """§1 single entry point: strict JSON, validated, retried, cached."""
        schema = schema if schema is not None else load_schema(task)
        system = load_prompt(task)
        user = json.dumps(payload, default=str, indent=2)

        # The provider is part of the cache key: the same filing classified by
        # a local 3B model and by Sonnet are different answers, and silently
        # serving one for the other would be a lie about provenance.
        content_hash = LLMCache.key(task, f"{self.provider}:{self.model}", payload)
        cache = self.cache
        if cache is not None:
            cached = cache.get(content_hash)
            if cached is not None:
                log.debug("LLM cache hit for %s (%s)", task, content_hash[:12])
                return Classification(
                    data=cached, task=task, from_cache=True,
                    latency_sec=0.0, model=self.model,
                )

        began = time.monotonic()
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = self._call_model(system, user, schema)
                data = extract_json(raw)
                validate(data, schema)
            except LLMSetupError:
                # Permanent: no server, no model, no key. Surface it now --
                # retrying cannot fix a configuration problem.
                raise
            except Exception as exc:
                last_error = exc
                log.warning("LLM %s attempt %d/%d failed: %s", task, attempt, self.retries, exc)
                if attempt < self.retries:
                    self._sleep(self.backoff_base * (2 ** (attempt - 1)))
                continue

            latency = time.monotonic() - began
            if cache is not None:
                cache.put(content_hash, task, self.model, data)
            return Classification(
                data=data, task=task, from_cache=False,
                latency_sec=latency, model=self.model, attempts=attempt,
            )

        raise LLMError(
            f"LLM task {task!r} failed after {self.retries} attempts: {last_error}"
        )


_default: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """Process-wide LLM client singleton."""
    global _default
    if _default is None:
        _default = LLMClient()
    return _default


def set_llm(client: Optional[LLMClient]) -> None:
    global _default
    _default = client


def classify(task: str, payload: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """§1: ``classify(task, payload, schema) -> dict``.

    This is the signature the spec names. Engines call it and get a validated
    dict back, or a loud :class:`LLMError`.
    """
    return get_llm().classify(task, payload, schema).data
