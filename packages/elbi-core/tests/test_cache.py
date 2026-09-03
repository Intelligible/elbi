"""Tests for cache policy and the cache stores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elbi_core import CacheError, cache
from elbi_core.artifact import Artifact
from elbi_core.cache import (
    CachedResult,
    CachePolicy,
    LocalCacheStore,
    MemoryCacheStore,
)


def _result(
    value: object = None, *, output_version: str = "ov", tags: tuple = ()
) -> CachedResult:
    rows = [{"a": 1}] if value is None else value
    return CachedResult(
        artifact=Artifact.table(rows)
        if isinstance(rows, list)
        else Artifact.json(rows),
        output_version=output_version,
        computed_at=1.0,
        ttl=None,
        tags=tags,
    )


# --- policy ----------------------------------------------------------------


def test_auto_defaults() -> None:
    policy = cache.auto()
    assert policy.enabled is True
    assert policy.deterministic is True
    assert policy.ttl is None


def test_auto_with_options() -> None:
    policy = cache.auto(ttl=60, deterministic=False, tags=["a", "b"])
    assert (policy.ttl, policy.deterministic, policy.tags) == (60, False, ("a", "b"))


def test_never() -> None:
    assert cache.never().enabled is False


def test_invalid_ttl_rejected() -> None:
    with pytest.raises(ValueError, match="ttl must be positive"):
        CachePolicy(ttl=0)


def test_expire_validation() -> None:
    with pytest.raises(ValueError, match="expire must be positive"):
        CachePolicy(expire=0)
    with pytest.raises(ValueError, match="expire must be >= ttl"):
        CachePolicy(ttl=100, expire=10)
    assert cache.auto(ttl=10, expire=100).expire == 100


# --- memory store ----------------------------------------------------------


def test_memory_store_roundtrip() -> None:
    store = MemoryCacheStore()
    assert store.get("dv") is None
    result = _result()
    store.put("dv", result)
    assert store.get("dv") is result


def test_memory_store_invalidate_tag_and_clear() -> None:
    store = MemoryCacheStore()
    store.put("a", _result(tags=("x",)))
    store.put("b", _result(tags=("y",)))
    assert store.invalidate_tag("x") == 1
    assert store.get("a") is None
    assert store.get("b") is not None
    store.clear()
    assert store.get("b") is None


# --- local on-disk store ---------------------------------------------------


def test_local_store_roundtrip(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path / "cache")
    assert store.get("dv") is None
    store.put("dv", _result(tags=("t",)))
    loaded = store.get("dv")
    assert loaded is not None
    assert loaded.artifact.value == [{"a": 1}]
    assert loaded.output_version == "ov"
    assert loaded.tags == ("t",)


def test_local_store_dedupes_blobs(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    store = LocalCacheStore(root)
    # Two different data versions with identical artifacts share one CAS blob.
    store.put("dv1", _result(output_version="ov1"))
    store.put("dv2", _result(output_version="ov2"))
    assert len(list((root / "cas").glob("*"))) == 1
    assert len(list((root / "ac").glob("*.json"))) == 2


def test_local_store_invalidate_tag(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path / "cache")
    store.put("a", _result(tags=("keep",)))
    store.put("b", _result(output_version="ov2", tags=("drop",)))
    assert store.invalidate_tag("drop") == 1
    assert store.get("b") is None
    assert store.get("a") is not None
    assert store.invalidate_tag("missing") == 0


def test_local_store_clear(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path / "cache")
    store.put("a", _result())
    store.clear()
    assert store.get("a") is None
    # Clearing again (nothing to remove) is safe.
    store.clear()


def test_local_store_corrupt_entry_raises(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    store = LocalCacheStore(root)
    store.put("dv", _result())
    (root / "ac" / "dv.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CacheError, match="corrupt cache entry"):
        store.get("dv")


def test_local_store_missing_blob_raises(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    store = LocalCacheStore(root)
    store.put("dv", _result())
    # Remove the CAS blob but keep the action-cache record.
    for blob in (root / "cas").glob("*"):
        blob.unlink()
    with pytest.raises(CacheError, match="corrupt cache entry"):
        store.get("dv")


def test_local_store_unpicklable_raises(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path / "cache")
    unpicklable = CachedResult(
        artifact=Artifact.json(lambda: 1),  # a lambda cannot be pickled
        output_version="ov",
        computed_at=1.0,
    )
    with pytest.raises(CacheError, match="not picklable"):
        store.put("dv", unpicklable)


def test_local_store_invalidate_tag_no_cache_dir(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path / "absent")
    assert store.invalidate_tag("x") == 0


def test_local_store_record_shape(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    LocalCacheStore(root).put("dv", _result(tags=("t",)))
    record = json.loads((root / "ac" / "dv.json").read_text())
    assert set(record) == {"output_version", "computed_at", "ttl", "tags", "blob"}


# --- HMAC signing (verify-before-unpickle) ---------------------------------


def test_local_store_tamper_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    store = LocalCacheStore(root)
    store.put("dv", _result())
    blob_file = next((root / "cas").glob("*"))
    data = bytearray(blob_file.read_bytes())
    data[-1] ^= 0xFF  # flip a byte in the pickled payload (after the MAC)
    blob_file.write_bytes(bytes(data))
    with pytest.raises(CacheError, match="signature mismatch"):
        store.get("dv")


def test_injected_secret_interop(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    LocalCacheStore(root, secret=b"k" * 32).put("dv", _result())
    # Same secret reads it; a different secret is rejected before unpickling.
    assert LocalCacheStore(root, secret=b"k" * 32).get("dv") is not None
    with pytest.raises(CacheError, match="signature mismatch"):
        LocalCacheStore(root, secret=b"x" * 32).get("dv")


def test_valid_signature_but_corrupt_payload(tmp_path: Path) -> None:
    import hashlib
    import hmac

    root = tmp_path / "cache"
    secret = b"k" * 32
    store = LocalCacheStore(root, secret=secret)
    store.put("dv", _result())
    record = json.loads((root / "ac" / "dv.json").read_text())
    # Correctly sign garbage so it passes the HMAC gate but fails to unpickle.
    garbage = b"not a pickle"
    signed = hmac.new(secret, garbage, hashlib.sha256).digest() + garbage
    (root / "cas" / record["blob"]).write_bytes(signed)
    with pytest.raises(CacheError, match="corrupt cache entry"):
        store.get("dv")


def test_local_key_file_is_private(tmp_path: Path) -> None:
    import os
    import stat

    root = tmp_path / "cache"
    LocalCacheStore(root).put("dv", _result())
    key_path = root / ".cache-key"
    assert key_path.exists()
    if os.name == "posix":  # Windows does not use POSIX permission bits
        assert stat.S_IMODE(key_path.stat().st_mode) & 0o077 == 0  # no group/other


# --- garbage collection ----------------------------------------------------


def test_derivation_tag() -> None:
    from elbi_core.cache import derivation_tag

    assert derivation_tag("revenue") == "derivation:revenue"


def test_invalidate_tag_reclaims_blobs(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path)
    store.put("dv", _result(tags=("t",)))
    assert store.invalidate_tag("t", grace=0) == 1  # record removed
    assert list((tmp_path / "cas").iterdir()) == []  # and its blob swept, not orphaned


def test_gc_reclaims_orphaned_blob(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path)
    store.put("dv", _result())
    (tmp_path / "ac" / "dv.json").unlink()  # orphan the blob
    assert store.gc(grace=0) == 1
    assert list((tmp_path / "cas").iterdir()) == []


def test_gc_keeps_referenced_blob(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path)
    store.put("dv", _result())
    assert store.gc(grace=0) == 0
    assert store.get("dv") is not None  # still intact


def test_gc_grace_spares_fresh_orphan(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path)
    store.put("dv", _result())
    (tmp_path / "ac" / "dv.json").unlink()
    assert store.gc() == 0  # default grace keeps a just-written orphan
    assert store.gc(grace=0) == 1  # forced sweep reclaims it


def test_gc_and_evict_on_empty_cache(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path / "empty")
    assert store.gc() == 0
    assert store.evict_older_than(10.0) == 0


def test_gc_with_blobs_but_no_action_cache(tmp_path: Path) -> None:
    # Blobs present with no action cache at all (e.g. after every record expired):
    # all blobs are unreferenced and reclaimable.
    cas = tmp_path / "cas"
    cas.mkdir(parents=True)
    (cas / "deadbeef").write_bytes(b"x")
    assert LocalCacheStore(tmp_path).gc(grace=0) == 1
    assert list(cas.iterdir()) == []


def test_evict_older_than_drops_aged_records(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path, clock=lambda: 1_000.0)
    store.put("dv", _result())  # computed_at=1.0, so ~999s old
    assert store.evict_older_than(10.0, grace=0) == 1
    assert store.get("dv") is None
    assert list((tmp_path / "cas").iterdir()) == []  # its blob swept too


def test_evict_older_than_keeps_fresh_records(tmp_path: Path) -> None:
    store = LocalCacheStore(tmp_path, clock=lambda: 5.0)
    store.put("dv", _result())  # computed_at=1.0, only 4s old
    assert store.evict_older_than(100.0, grace=0) == 0
    assert store.get("dv") is not None
