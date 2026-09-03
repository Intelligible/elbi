"""Audit trail of derivation invocations.

The serving layer emits one :class:`AuditEvent` per invocation to a pluggable
:class:`AuditSink`. :class:`NullAuditSink` (the default) discards events;
:class:`JsonlAuditSink` appends them to a file. :meth:`AuditEvent.as_record` keys
follow the OpenTelemetry MCP semantic conventions where they overlap.

:class:`ChainedJsonlAuditSink` adds tamper-evidence: each line carries a sequence
number and the hash of the previous line, so editing, reordering, or dropping a
record breaks the chain. :func:`verify_chain` checks it offline and returns the
head hash, which an operator can anchor externally to close the one gap a chain
cannot see on its own: truncation of the newest records.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is POSIX-only; a no-op on Windows
    fcntl = None  # type: ignore[assignment]

from .versioning import hash_bytes


@dataclass(frozen=True)
class AuditEvent:
    """A single derivation invocation.

    ``version`` is the data version that produced the answer (``None`` when the
    derivation is uncached); ``outcome`` is the execution outcome (``"ok"`` or
    ``"error"``).
    """

    derivation: str
    timestamp: float
    outcome: str
    version: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_record(self) -> dict[str, Any]:
        """A JSON-serializable record with OTel-MCP-aligned keys."""
        record: dict[str, Any] = {
            "timestamp": self.timestamp,
            "mcp.tool.name": self.derivation,
            "elbi.outcome": self.outcome,
        }
        if self.version is not None:
            record["elbi.derivation.version"] = self.version
        if self.params:
            record["elbi.params"] = dict(self.params)
        if self.error is not None:
            record["error.type"] = self.error
        return record


@runtime_checkable
class AuditSink(Protocol):
    """A destination for audit events. Implementations MUST be thread-safe."""

    def record(self, event: AuditEvent) -> None:
        """Persist one audit event."""
        ...


class NullAuditSink:
    """Drop every event. The local default: no trail unless one is configured."""

    def record(self, event: AuditEvent) -> None:
        """Discard ``event``."""


class JsonlAuditSink:
    """Append each event to a file as one JSON object per line.

    Writes are append-only and lock-guarded, so concurrent serves produce an
    ordered trail.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        """Append ``event`` to the log as a single JSON line."""
        line = json.dumps(event.as_record(), default=str, separators=(",", ":"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


#: The SHA-256 slot for the record before the first one: a chain's anchor.
GENESIS_HASH = "0" * 64
#: The per-record chain keys added by :class:`ChainedJsonlAuditSink`.
SEQ_KEY = "elbi.audit.seq"
PREV_KEY = "elbi.audit.prev"


class ChainedJsonlAuditSink:
    """Append events as a hash-chained JSON-lines log.

    Each line is :meth:`AuditEvent.as_record` plus a 0-based ``seq`` and ``prev``,
    the SHA-256 of the previous raw line's bytes (``GENESIS_HASH`` for the first).
    The line's own bytes are the artifact hashed forward, so any edit to line ``n``
    changes its hash and breaks line ``n + 1``'s ``prev``. A file written by the
    plain :class:`JsonlAuditSink` may precede the chained records; the chain simply
    starts forward from the last existing line. Thread-safe within one process; an
    advisory file lock (POSIX only) serializes each append across processes too, so
    two processes never interleave and corrupt a line, but each still tracks its own
    sequence counter in memory, so concurrent writer processes can still fork the
    chain's sequence numbers even though neither can corrupt the other's bytes.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._next_seq, self._prev = _tail_state(self._path)

    @property
    def path(self) -> Path:
        """The log file this sink appends to."""
        return self._path

    def record(self, event: AuditEvent) -> None:
        """Append ``event`` as a chained JSON line."""
        with self._lock:
            record = event.as_record()
            record[SEQ_KEY] = self._next_seq
            record[PREV_KEY] = self._prev
            line = json.dumps(record, default=str, separators=(",", ":"))
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                handle.write(line + "\n")
                handle.flush()
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            self._next_seq += 1
            self._prev = hash_bytes(line.encode("utf-8"))


@dataclass(frozen=True)
class ChainReport:
    """The result of verifying an audit-log chain.

    ``head`` is the SHA-256 of the last line, the value to anchor externally: a
    chain cannot detect truncation of its own newest records, so a recorded head
    that no longer matches reveals a dropped tail.
    """

    ok: bool
    records: int
    chained: int
    head: str | None
    error: str | None = None


#: Bytes read from EOF before falling back to a full scan.
_TAIL_BLOCK_BYTES = 65536


def _tail_state(path: Path) -> tuple[int, str]:
    """The next sequence number and previous-line hash to resume a chain.

    Reads only the last ``_TAIL_BLOCK_BYTES`` of the file first; falls back to a
    full scan only when that window holds no chained record at all.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0, GENESIS_HASH
    if size == 0:
        return 0, GENESIS_HASH
    with path.open("rb") as handle:
        handle.seek(max(0, size - _TAIL_BLOCK_BYTES))
        tail_bytes = handle.read()
    if size > _TAIL_BLOCK_BYTES:
        # Reading from a mid-file offset may have started inside a line; that
        # partial leading fragment is not a real line, so drop it.
        tail_bytes = tail_bytes.split(b"\n", 1)[-1] if b"\n" in tail_bytes else b""
    tail_lines = [
        line for line in tail_bytes.decode("utf-8", "replace").splitlines() if line
    ]
    if tail_lines:
        resumed = _resume_state(tail_lines)
        if resumed is not None:
            return resumed
    lines = _nonempty_lines(path)
    if not lines:
        return 0, GENESIS_HASH
    return _resume_state(lines) or (len(lines), hash_bytes(lines[-1].encode("utf-8")))


def _resume_state(lines: list[str]) -> tuple[int, str] | None:
    """The ``(next_seq, prev_hash)`` to resume from, or ``None`` if none is chained."""
    prev = hash_bytes(lines[-1].encode("utf-8"))
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - defensive on a bad tail
            continue
        seq = record.get(SEQ_KEY) if isinstance(record, dict) else None
        if isinstance(seq, int):
            return seq + 1, prev
    return None


def _nonempty_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line for line in text.splitlines() if line]


def verify_chain(path: str | Path) -> ChainReport:
    """Verify an audit-log chain offline, reporting the first inconsistency.

    A leading run of unchained records (a legacy log) is allowed; once a chained
    record appears, every later record must be chained, its ``prev`` must equal the
    previous raw line's hash, and its ``seq`` must increment by one. Missing or
    malformed files with no records verify as an empty chain. Never raises for
    content problems: the outcome is the report's ``ok`` flag.

    Streams the file line by line rather than materializing it: the walk only ever
    needs the current line and the previous line's hash, and a broken chain stops
    reading at the first inconsistency instead of scanning to the true end of file.
    On a broken chain, ``head`` is the hash of the line where verification stopped,
    not necessarily the file's true last line.
    """
    if not Path(path).exists():
        return ChainReport(ok=True, records=0, chained=0, head=None)
    with Path(path).open(encoding="utf-8") as handle:
        previous_line_hash = GENESIS_HASH
        prev_seq: int | None = None
        records = 0
        chained = 0
        seen_chained = False
        head: str | None = None
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            records += 1
            head = hash_bytes(line.encode("utf-8"))
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return ChainReport(
                    ok=False,
                    records=records,
                    chained=chained,
                    head=head,
                    error=f"line {records}: not valid JSON",
                )
            is_chained = isinstance(record, dict) and (
                SEQ_KEY in record or PREV_KEY in record
            )
            if seen_chained and not is_chained:
                return ChainReport(
                    ok=False,
                    records=records,
                    chained=chained,
                    head=head,
                    error=f"line {records}: chain ends early",
                )
            if is_chained:
                error = _check_link(record, records, previous_line_hash, prev_seq)
                if error is not None:
                    return ChainReport(
                        ok=False,
                        records=records,
                        chained=chained,
                        head=head,
                        error=error,
                    )
                prev_seq = record[SEQ_KEY]
                seen_chained = True
                chained += 1
            previous_line_hash = head
    return ChainReport(ok=True, records=records, chained=chained, head=head)


def _check_link(
    record: dict[str, Any], number: int, previous_line_hash: str, prev_seq: int | None
) -> str | None:
    """The reason a chained record breaks the chain, or ``None`` if it holds."""
    if SEQ_KEY not in record or PREV_KEY not in record:
        return f"line {number}: incomplete chain fields"
    if not isinstance(record[SEQ_KEY], int):
        return f"line {number}: sequence is not an integer"
    if record[PREV_KEY] != previous_line_hash:
        return f"line {number}: previous-hash mismatch"
    if prev_seq is not None and record[SEQ_KEY] != prev_seq + 1:
        return f"line {number}: sequence gap"
    return None
