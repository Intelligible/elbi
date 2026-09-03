"""Tests for the audit trail: events and sinks."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from elbi_core import (
    AuditEvent,
    AuditSink,
    ChainedJsonlAuditSink,
    JsonlAuditSink,
    NullAuditSink,
    verify_chain,
)
from elbi_core.audit import _TAIL_BLOCK_BYTES, GENESIS_HASH, PREV_KEY, SEQ_KEY


def test_event_record_omits_absent_optional_fields() -> None:
    event = AuditEvent(derivation="revenue", timestamp=1.0, outcome="ok")
    record = event.as_record()
    assert record == {
        "timestamp": 1.0,
        "mcp.tool.name": "revenue",
        "elbi.outcome": "ok",
    }


def test_event_record_includes_present_optional_fields() -> None:
    event = AuditEvent(
        derivation="revenue",
        timestamp=2.0,
        outcome="error",
        version="abc123",
        params={"zip": "94103"},
        error="AuthorizationError",
    )
    record = event.as_record()
    assert record["elbi.derivation.version"] == "abc123"
    assert record["elbi.params"] == {"zip": "94103"}
    assert record["error.type"] == "AuthorizationError"


def test_null_sink_discards() -> None:
    sink: AuditSink = NullAuditSink()
    sink.record(AuditEvent("x", 1.0, "ok"))  # no error, nothing persisted


def test_jsonl_sink_appends_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "audit.jsonl"  # parent dir does not exist yet
    sink = JsonlAuditSink(path)
    sink.record(AuditEvent("a", 1.0, "ok", version="v1"))
    sink.record(AuditEvent("b", 2.0, "error", error="AuthorizationError"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["mcp.tool.name"] == "a"
    assert first["elbi.derivation.version"] == "v1"
    assert json.loads(lines[1])["error.type"] == "AuthorizationError"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_chain(path: Path, count: int) -> ChainedJsonlAuditSink:
    sink = ChainedJsonlAuditSink(path)
    for i in range(count):
        sink.record(AuditEvent(f"d{i}", float(i), "allow", "ok"))
    return sink


def test_chain_genesis_and_links(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 3)
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first[SEQ_KEY] == 0
    assert first[PREV_KEY] == GENESIS_HASH
    # Each record's prev is the SHA-256 of the previous raw line, recomputed here.
    second = json.loads(lines[1])
    assert second[SEQ_KEY] == 1
    assert second[PREV_KEY] == _sha256(lines[0])
    assert json.loads(lines[2])[PREV_KEY] == _sha256(lines[1])


def test_verify_chain_ok_and_head(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 4)
    report = verify_chain(path)
    assert report.ok
    assert report.records == 4
    assert report.chained == 4
    head = _sha256(path.read_text(encoding="utf-8").splitlines()[-1])
    assert report.head == head


def test_verify_chain_empty_and_missing(tmp_path: Path) -> None:
    missing = verify_chain(tmp_path / "nope.jsonl")
    assert missing.ok and missing.records == 0 and missing.head is None
    assert missing.chained == 0
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    report = verify_chain(empty)
    assert report.ok and report.records == 0 and report.chained == 0


def test_sink_creates_missing_parent_dirs(tmp_path: Path) -> None:
    # The sink must create the whole parent chain, not just one level.
    path = tmp_path / "a" / "b" / "audit.jsonl"
    ChainedJsonlAuditSink(path).record(AuditEvent("d", 0.0, "ok"))
    assert path.exists()
    assert verify_chain(path).ok


def _append_line(path: Path, record: dict[str, object]) -> None:
    line = json.dumps(record, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def test_forged_record_with_low_or_high_prev_breaks(tmp_path: Path) -> None:
    # A forged record whose prev is a valid-looking hash but not the real previous
    # line's must fail regardless of how that hash sorts against the true one. Two
    # extremes (all-zero and all-f) pin the equality check against ordering mutants.
    for forged_prev in (GENESIS_HASH, "f" * 64):
        path = tmp_path / f"audit-{forged_prev[0]}.jsonl"
        _write_chain(path, 2)
        last_seq = json.loads(path.read_text().splitlines()[-1])[SEQ_KEY]
        _append_line(
            path,
            {
                "timestamp": 9.0,
                "mcp.tool.name": "forged",
                "elbi.outcome": "ok",
                SEQ_KEY: last_seq + 1,
                PREV_KEY: forged_prev,  # a wrong prev, whichever way it sorts
            },
        )
        report = verify_chain(path)
        assert not report.ok, f"prev={forged_prev[0]} not detected"


def test_seq_gap_breaks_either_direction(tmp_path: Path) -> None:
    # A record with the correct prev-hash but a wrong seq (too small or too large)
    # must break: the seq must increment by exactly one, not merely differ one way.
    for wrong_seq in (0, 99):
        path = tmp_path / f"audit-{wrong_seq}.jsonl"
        _write_chain(path, 2)
        real_prev = _sha256(path.read_text().splitlines()[-1])
        _append_line(
            path,
            {
                "timestamp": 9.0,
                "mcp.tool.name": "gap",
                "elbi.outcome": "ok",
                SEQ_KEY: wrong_seq,  # not prev_seq + 1
                PREV_KEY: real_prev,  # the hash IS correct, so only seq is wrong
            },
        )
        assert not verify_chain(path).ok, f"seq={wrong_seq} not detected"


def test_restart_after_three_continues_sequence(tmp_path: Path) -> None:
    # Resuming after 3 records must set the next seq to 3, not double it: an off-by
    # arithmetic slip would gap the chain. (Two records hide it: 1+1 == 1<<1.)
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 3)
    ChainedJsonlAuditSink(path).record(AuditEvent("d3", 3.0, "ok"))
    report = verify_chain(path)
    assert report.ok and report.records == 4
    seqs = [
        json.loads(line)[SEQ_KEY]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert seqs == [0, 1, 2, 3]


def test_resume_reads_only_the_tail_on_a_large_log(tmp_path: Path) -> None:
    # Enough records to exceed _TAIL_BLOCK_BYTES, so resuming exercises the bounded
    # tail read (a mid-file seek and partial-line trim), not the small-file whole-read
    # path every other resume test in this file happens to take.
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 1000)
    assert path.stat().st_size > _TAIL_BLOCK_BYTES

    resumed = ChainedJsonlAuditSink(path)
    resumed.record(AuditEvent("new", 1000.0, "ok"))

    report = verify_chain(path)
    assert report.ok and report.records == 1001
    last_line = path.read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(last_line)[SEQ_KEY] == 1000


def test_flipped_byte_breaks_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 3)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"allow"', '"deny"')  # tamper line 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = verify_chain(path)
    assert not report.ok
    assert "line 2" in (report.error or "")  # line 2's prev no longer matches line 1


def test_broken_chain_head_is_the_line_where_verification_stopped(
    tmp_path: Path,
) -> None:
    # verify_chain streams the file and stops at the first break, so on a broken
    # chain head is the hash of the offending line, not the file's true last line
    # (which the old, list-based implementation reported as a byproduct of reading
    # the whole file upfront).
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 4)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"allow"', '"deny"')  # tamper line 1, breaks at line 2
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = verify_chain(path)
    assert not report.ok and "line 2" in (report.error or "")
    assert report.head == _sha256(lines[1])
    assert report.head != _sha256(lines[3])  # not the file's true tail


def test_deleted_line_breaks_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 4)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # drop a middle record
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_chain(path).ok


def test_swapped_lines_break_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 3)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_chain(path).ok


def test_non_json_line_breaks_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 2)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    report = verify_chain(path)
    assert not report.ok and "line 3" in (report.error or "")


def test_tail_truncation_is_undetectable_by_construction(tmp_path: Path) -> None:
    # A chain cannot see records dropped from its own tail; the head hash is what an
    # operator anchors externally to catch it. This encodes that known limitation.
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 4)
    full_head = verify_chain(path).head
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    truncated = verify_chain(path)
    assert truncated.ok  # still internally consistent
    assert truncated.head != full_head  # but the head moved: the anchor catches it


def test_restart_continues_one_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 2)
    # A fresh sink instance over the same file resumes the sequence and links forward.
    resumed = ChainedJsonlAuditSink(path)
    resumed.record(AuditEvent("d2", 2.0, "ok"))
    report = verify_chain(path)
    assert report.ok and report.records == 3
    assert json.loads(path.read_text().splitlines()[2])[SEQ_KEY] == 2


def test_legacy_prefix_then_chained_verifies(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    JsonlAuditSink(path).record(AuditEvent("legacy", 0.0, "ok"))
    ChainedJsonlAuditSink(path).record(AuditEvent("new", 1.0, "ok"))
    report = verify_chain(path)
    assert report.ok and report.records == 2 and report.chained == 1


def test_plain_line_after_chained_breaks(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write_chain(path, 2)
    JsonlAuditSink(path).record(AuditEvent("plain", 9.0, "ok"))
    report = verify_chain(path)
    assert not report.ok and "ends early" in (report.error or "")


def test_concurrent_writers_produce_one_valid_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = ChainedJsonlAuditSink(path)

    def worker(base: int) -> None:
        for i in range(25):
            sink.record(AuditEvent(f"d{base}-{i}", float(i), "allow", "ok"))

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report = verify_chain(path)
    assert report.ok and report.records == 200
    seqs = sorted(
        json.loads(line)[SEQ_KEY]
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert seqs == list(range(200))
