"""Caching: per-derivation policy and the stores that persist outputs.

The policy is :func:`auto` (content-addressed; reuse while code, params, and
inputs are unchanged) or :func:`never` (always recompute). ``ttl`` adds
time-bounded staleness; ``deterministic=False`` marks outputs whose bytes are
unstable for identical inputs (e.g. model training). Those are still cached but
never used for early cutoff.

A :class:`CacheStore` is the pluggable backend: an in-memory session store and an
on-disk store split into an action cache (data version → metadata) and a
content-addressable store (deduplicated, pickled artifact blobs).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import secrets
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .artifact import Artifact
from .errors import CacheError
from .versioning import hash_bytes

_MAC_LEN = 32  # HMAC-SHA256 digest size in bytes
#: Default grace before an unreferenced blob is swept, so a blob written by a
#: concurrent process (blob first, record next) is not reclaimed mid-write.
_GC_GRACE = 3600.0


def derivation_tag(name: str) -> str:
    """The cache tag a derivation's entries carry, so they invalidate by name.

    The action cache is keyed by content hash, not by name; tagging every entry
    with this gives a stable handle to purge one derivation's cache on delete.
    """
    return f"derivation:{name}"


@dataclass(frozen=True)
class CachePolicy:
    """How a derivation's output is cached and invalidated.

    Within ``ttl`` a cached value is served as-is; past ``ttl`` it is served stale
    while refreshing in the background; past ``expire`` the next read blocks for a
    fresh recompute and never serves something older. ``expire`` bounds
    stale-if-error so a perpetually-failing refresh cannot serve stale forever.
    """

    enabled: bool = True
    ttl: float | None = None
    expire: float | None = None
    deterministic: bool = True
    code_version: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.ttl is not None and self.ttl <= 0:
            raise ValueError("cache ttl must be positive")
        if self.expire is not None:
            if self.expire <= 0:
                raise ValueError("cache expire must be positive")
            if self.ttl is not None and self.expire < self.ttl:
                raise ValueError("cache expire must be >= ttl")


def auto(
    *,
    ttl: float | None = None,
    expire: float | None = None,
    deterministic: bool = True,
    code_version: str | None = None,
    tags: list[str] | tuple[str, ...] = (),
) -> CachePolicy:
    """Content-addressed caching (the default policy)."""
    return CachePolicy(
        enabled=True,
        ttl=ttl,
        expire=expire,
        deterministic=deterministic,
        code_version=code_version,
        tags=tuple(tags),
    )


def never() -> CachePolicy:
    """Disable caching: always recompute. For derivations with untracked inputs."""
    return CachePolicy(enabled=False)


#: The default policy applied when a derivation declares none.
DEFAULT = CachePolicy()


@dataclass(frozen=True)
class CachedResult:
    """A cached derivation output plus the metadata needed to validate it."""

    artifact: Artifact
    output_version: str
    computed_at: float
    ttl: float | None = None
    tags: tuple[str, ...] = ()


class CacheStore(Protocol):
    """A backend that persists and retrieves derivation outputs by data version."""

    def get(self, data_version: str) -> CachedResult | None:
        """Return the cached result for ``data_version``, or None if absent."""
        ...

    def put(self, data_version: str, result: CachedResult) -> None:
        """Store ``result`` under ``data_version``."""
        ...

    def invalidate_tag(self, tag: str) -> int:
        """Remove all entries carrying ``tag``; return how many were removed."""
        ...

    def clear(self) -> None:
        """Remove every entry."""
        ...


class MemoryCacheStore:
    """An in-process cache store: the per-session layer, and handy in tests."""

    def __init__(self) -> None:
        self._entries: dict[str, CachedResult] = {}

    def get(self, data_version: str) -> CachedResult | None:
        """Return the in-memory result for ``data_version``, or None."""
        return self._entries.get(data_version)

    def put(self, data_version: str, result: CachedResult) -> None:
        """Store ``result`` in memory under ``data_version``."""
        self._entries[data_version] = result

    def invalidate_tag(self, tag: str) -> int:
        """Drop in-memory entries carrying ``tag``; return the count removed."""
        removed = [k for k, v in self._entries.items() if tag in v.tags]
        for key in removed:
            del self._entries[key]
        return len(removed)

    def clear(self) -> None:
        """Drop all in-memory entries."""
        self._entries.clear()


@dataclass(frozen=True)
class _Record:
    output_version: str
    computed_at: float
    ttl: float | None
    tags: list[str]
    blob: str


class LocalCacheStore:
    """An on-disk cache store: an action cache plus a content-addressable store.

    Blobs are HMAC-signed and verified before unpickling, so a tampered or foreign
    blob is rejected rather than executed. ``secret`` supplies the key; when
    omitted, a per-cache 256-bit key is generated and stored ``0600`` under the
    cache root. A shared or remote store must pass an injected ``secret``, since
    auto-generated local keys are not portable across hosts.
    """

    def __init__(
        self,
        root: Path,
        *,
        secret: bytes | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._root = root
        self._ac = root / "ac"
        self._cas = root / "cas"
        self._secret = secret
        self._clock = clock if clock is not None else time.time

    def get(self, data_version: str) -> CachedResult | None:
        """Load the on-disk result for ``data_version``, or None if absent."""
        record_path = self._ac / f"{data_version}.json"
        if not record_path.exists():
            return None
        try:
            record = _Record(**json.loads(record_path.read_text(encoding="utf-8")))
            signed = (self._cas / record.blob).read_bytes()
            mac, blob = signed[:_MAC_LEN], signed[_MAC_LEN:]
        except (OSError, ValueError, TypeError) as exc:
            raise CacheError(f"corrupt cache entry for {data_version}: {exc}") from exc
        # Verify the signature before unpickling; never unpickle unsigned bytes.
        if not hmac.compare_digest(mac, self._sign(blob)):
            raise CacheError(
                f"cache signature mismatch for {data_version}; refusing to unpickle"
            )
        try:
            artifact = pickle.loads(blob)  # noqa: S301 - signature-verified above
        except (pickle.UnpicklingError, ValueError, TypeError, EOFError) as exc:
            raise CacheError(f"corrupt cache entry for {data_version}: {exc}") from exc
        if not isinstance(artifact, Artifact):  # pragma: no cover - defensive
            raise CacheError(f"cache blob for {data_version} is not an Artifact")
        return CachedResult(
            artifact=artifact,
            output_version=record.output_version,
            computed_at=record.computed_at,
            ttl=record.ttl,
            tags=tuple(record.tags),
        )

    def put(self, data_version: str, result: CachedResult) -> None:
        """Write ``result`` to the action cache + content-addressable store."""
        self._ac.mkdir(parents=True, exist_ok=True)
        self._cas.mkdir(parents=True, exist_ok=True)
        try:
            blob = pickle.dumps(result.artifact, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            raise CacheError(
                "derivation output is not picklable and cannot be cached; "
                "declare cache.never() if it holds an unpicklable value"
            ) from exc
        digest = hash_bytes(blob)
        blob_path = self._cas / digest
        if not blob_path.exists():
            _atomic_write_bytes(blob_path, self._sign(blob) + blob)
        record = _Record(
            output_version=result.output_version,
            computed_at=result.computed_at,
            ttl=result.ttl,
            tags=list(result.tags),
            blob=digest,
        )
        _atomic_write_text(
            self._ac / f"{data_version}.json", json.dumps(record.__dict__)
        )

    def invalidate_tag(self, tag: str, *, grace: float = _GC_GRACE) -> int:
        """Delete action-cache entries carrying ``tag``, then sweep their blobs.

        Returns the number of action-cache records removed. The blob sweep
        reclaims content no surviving record references (deferred by ``grace``).
        """
        if not self._ac.exists():
            return 0
        removed = 0
        for record_path in self._ac.glob("*.json"):
            try:
                tags = json.loads(record_path.read_text(encoding="utf-8")).get(
                    "tags", []
                )
            except (OSError, ValueError):  # pragma: no cover - defensive
                continue
            if tag in tags:
                record_path.unlink(missing_ok=True)
                removed += 1
        self._sweep_cas(grace=grace)
        return removed

    def gc(self, *, grace: float = _GC_GRACE) -> int:
        """Reclaim blobs no action-cache record references; return how many.

        A mark-and-sweep over the action cache (the roots) and the blob store, so
        orphans left by deletes, evictions, or crashes are collected. ``grace``
        spares blobs written within the window, so a concurrent in-flight write
        (blob first, record next) is not reclaimed before its record lands.
        """
        return self._sweep_cas(grace=grace)

    def evict_older_than(self, max_age: float, *, grace: float = _GC_GRACE) -> int:
        """Drop action-cache records older than ``max_age`` seconds, then sweep.

        Age is measured from each record's ``computed_at``. Returns the number of
        records evicted. A backstop against unbounded growth in a long-running
        autonomous loop.
        """
        if not self._ac.exists():
            return 0
        now = self._clock()
        removed = 0
        for record_path in self._ac.glob("*.json"):
            try:
                computed_at = json.loads(record_path.read_text(encoding="utf-8"))[
                    "computed_at"
                ]
            except (OSError, ValueError, KeyError):  # pragma: no cover - defensive
                continue
            if now - float(computed_at) > max_age:
                record_path.unlink(missing_ok=True)
                removed += 1
        self._sweep_cas(grace=grace)
        return removed

    def _sweep_cas(self, *, grace: float) -> int:
        """Delete blobs unreferenced by any record and older than ``grace``."""
        if not self._cas.exists():
            return 0
        referenced: set[str] = set()
        if self._ac.exists():
            for record_path in self._ac.glob("*.json"):
                try:
                    referenced.add(
                        json.loads(record_path.read_text(encoding="utf-8"))["blob"]
                    )
                except (OSError, ValueError, KeyError):  # pragma: no cover - defensive
                    continue
        # Blob mtimes are real wall-clock, so the grace window is measured against
        # real time, independent of any injected logical clock (used for record age).
        now = time.time()
        removed = 0
        for blob_path in self._cas.iterdir():
            if blob_path.name in referenced:
                continue
            if now - blob_path.stat().st_mtime >= grace:
                blob_path.unlink(missing_ok=True)
                removed += 1
        return removed

    def clear(self) -> None:
        """Remove the entire on-disk cache (action cache + blobs)."""
        for path in (self._ac, self._cas):
            if path.exists():
                shutil.rmtree(path)

    def _sign(self, blob: bytes) -> bytes:
        return hmac.new(self._key(), blob, hashlib.sha256).digest()

    def _key(self) -> bytes:
        if self._secret is not None:
            return self._secret
        key_path = self._root / ".cache-key"
        try:
            return key_path.read_bytes()
        except FileNotFoundError:
            self._root.mkdir(parents=True, exist_ok=True)
            key = secrets.token_bytes(32)
            try:  # exclusive create; lose the race → read the winner's key
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:  # pragma: no cover - concurrent first-write race
                return key_path.read_bytes()
            with os.fdopen(fd, "wb") as handle:
                handle.write(key)
            return key


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
