"""
Simple disk-based cache for LLM calls, keyed by a hash of the function
name + serialized inputs. Meant to eliminate repeated Gemini calls when
re-running the same test contract during development -- NOT a production
cache (no TTL, no invalidation beyond deleting the cache folder).

On by default. Turn off with LLM_CACHE_ENABLED=false in .env if you want
every run to hit the API fresh (e.g. once you're past dev/testing).

Usage:
    from llm_cache import cached_call

    result = cached_call(
        "extract_clause", inputs_dict,
        lambda: chain.invoke(inputs_dict),
        ClauseExtraction,
    )
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".llm_cache"))
CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"

T = TypeVar("T", bound=BaseModel)

_stats = {"hits": 0, "misses": 0}


def reset_stats() -> None:
    """Call at the start of a request to get per-request hit/miss counts."""
    _stats["hits"] = 0
    _stats["misses"] = 0


def get_stats() -> dict:
    return dict(_stats)


def _cache_key(name: str, inputs: dict) -> str:
    payload = json.dumps({"name": name, "inputs": inputs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cached_call(name: str, inputs: dict, compute: Callable[[], T], model_cls: type[T]) -> T:
    """
    name: a short label for the calling function (e.g. "extract_clause")
    inputs: the dict of inputs that determines the output -- used for hashing.
            Must be JSON-serializable (strings/lists/dicts).
    compute: a zero-arg function that actually calls the LLM if needed.
    model_cls: the Pydantic class of the expected result, for deserializing
               a cache hit.
    """
    if not CACHE_ENABLED:
        return compute()

    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(name, inputs)
    path = CACHE_DIR / f"{key}.json"

    if path.exists():
        _stats["hits"] += 1
        data = json.loads(path.read_text(encoding="utf-8"))
        return model_cls.model_validate(data)

    _stats["misses"] += 1
    result = compute()
    path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result