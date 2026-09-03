"""Unit tests for the harness plumbing: scoring, report, loader, and the LLM baseline.

These use fabricated traps and a fake LLM client (no network), so the harness logic is
checked independently of the real gates and of any API key.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from benchmark import llm_baseline
from benchmark.harness import score
from benchmark.report import render_markdown, to_json
from benchmark.trap import Trap


@dataclass
class _FakeReport:
    verdict: str
    text: str

    def render(self) -> str:
        return self.text


def _trap(
    *,
    verdict: str,
    text: str,
    expected_pivotal: str | None = None,
    is_real: bool = False,
) -> Trap:
    report = _FakeReport(verdict, text)
    return Trap(
        id="fake_trap",
        pitfall="confounding",
        description="a fabricated trap",
        provenance="synthetic; test",
        added_by="tester",
        reviewed_by=("tester",),
        status="candidate",
        gate="verify_all",
        claim={"x": "a", "y": "b"},
        expected_verdict="unsound",
        expected_pivotal=expected_pivotal,
        rationale="test rationale",
        naive_verdict="sound",
        naive_rationale="looks causal",
        bite_module="effect.py",
        data=lambda: [{"a": "1", "b": "2"}],
        verify=lambda rows: report,
        is_real=is_real,
        fixture="traps/fake_trap/manifest.json",
    )


def test_score_passes_when_verdict_and_pivotal_match() -> None:
    trap = _trap(
        verdict="unsound",
        text="pivotal: z is a confounder",
        expected_pivotal="confounder",
    )
    result = score(trap, {})
    assert result.passed
    assert result.beats_llm is None and result.warning is None and not result.saturated


def test_score_fails_on_wrong_verdict() -> None:
    assert not score(_trap(verdict="sound", text="all good"), {}).passed


def test_score_fails_when_pivotal_missing() -> None:
    trap = _trap(
        verdict="unsound", text="something else", expected_pivotal="confounder"
    )
    assert not score(trap, {}).passed


def test_beats_llm_when_gate_disagrees_with_capture() -> None:
    trap = _trap(
        verdict="unsound", text="pivotal: confounder", expected_pivotal="confounder"
    )
    result = score(trap, {"fake_trap": {"verdict": "sound", "model": "m"}})
    assert result.beats_llm is True and not result.saturated and result.warning is None


def test_saturated_when_llm_matches_gate() -> None:
    trap = _trap(
        verdict="unsound", text="pivotal: confounder", expected_pivotal="confounder"
    )
    result = score(trap, {"fake_trap": {"verdict": "unsound", "model": "m"}})
    assert result.beats_llm is False and result.saturated
    assert result.warning is not None and "no gate advantage" in result.warning


def test_score_treats_malformed_capture_as_uncaptured() -> None:
    # The baseline is report-only; a missing/garbage capture must not crash the run.
    trap = _trap(
        verdict="unsound", text="pivotal: confounder", expected_pivotal="confounder"
    )
    for bad in ({}, {"model": "m"}, {"verdict": 123}, {"verdict": "banana"}):
        result = score(trap, {"fake_trap": bad})
        assert result.llm_verdict is None and result.beats_llm is None
        assert not result.saturated and result.passed


def test_report_json_and_markdown() -> None:
    trap = _trap(
        verdict="unsound", text="pivotal: confounder", expected_pivotal="confounder"
    )
    results = [score(trap, {"fake_trap": {"verdict": "unsound", "model": "m"}})]
    doc = to_json(results)
    assert doc["summary"]["total"] == 1 and doc["summary"]["passed"] == 1
    assert doc["summary"]["saturated"] == 1
    assert doc["summary"]["llm_models"] == ["m"]
    assert doc["flagged_for_pruning"] == ["fake_trap"]
    md = render_markdown(results)
    assert "# Statistical-pitfall benchmark" in md and "PASS" in md
    assert "Flagged for pruning" in md


# --- loader: the file-backed (real-trap) data path stays live without real data ---


def test_read_file_rows_parses_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from benchmark.loader import _read_file_rows

    path = tmp_path / "data.jsonl"
    path.write_text('{"x": 1, "y": "a"}\n{"x": 2, "y": "b"}\n', encoding="utf-8")
    rows = _read_file_rows(path)
    assert rows == [{"x": "1", "y": "a"}, {"x": "2", "y": "b"}]


def test_data_builder_reads_a_file_backed_trap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from benchmark.loader import _data_builder

    (tmp_path / "data.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    build = _data_builder({"file": "data.jsonl"}, tmp_path)
    assert build() == [{"x": "1"}]


def test_generate_rows_from_generated_by() -> None:
    # The authoring tool that produces a synthetic trap's committed data.jsonl.
    from benchmark.loader import generate_rows

    rows = generate_rows({"generated_by": {"generator": "noise", "seed": 0, "n": 5}})
    assert len(rows) == 5 and set(rows[0]) == {"x", "y"}


# --- the bare-LLM baseline plumbing (fake client, no network) ------------------


class _FakeClient:
    def __init__(self, text: str) -> None:
        self._text = text

    def step(self, transcript: object, tools: object) -> object:
        from elbi_agent.llm import Step

        return Step(text=self._text, end=True)


def _fake_trap() -> Trap:
    return _trap(verdict="unsound", text="x")


def test_query_llm_parses_the_final_verdict() -> None:
    verdict, excerpt = llm_baseline.query_llm(
        _FakeClient("reasoning\nVERDICT: sound"), _fake_trap(), [{"a": "1", "b": "2"}]
    )
    assert verdict == "sound" and excerpt


def test_query_llm_defaults_to_inconclusive_on_garbage() -> None:
    verdict, _ = llm_baseline.query_llm(
        _FakeClient("no verdict here"), _fake_trap(), [{"a": "1"}]
    )
    assert verdict == "inconclusive"


def test_load_baseline_missing_file_is_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert llm_baseline.load_baseline(tmp_path / "nope.json") == {}


def test_dump_data_materialises_every_trap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from benchmark.loader import load_traps
    from benchmark.run import main

    assert main(["--dump-data", str(tmp_path)]) == 0
    for trap in load_traps():
        assert (tmp_path / f"{trap.id}.csv").exists()


def test_refresh_without_anthropic_key_skips(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from benchmark.run import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--refresh-llm"]) == 2


def test_make_client_defaults_to_sonnet_5() -> None:
    from benchmark.run import DEFAULT_MODEL

    assert DEFAULT_MODEL == "claude-sonnet-5"


def test_make_client_litellm_requires_model() -> None:
    from benchmark.run import _make_client

    with pytest.raises(SystemExit):
        _make_client("litellm", None)


def test_run_benchmark_reports_progress_per_trap() -> None:
    from benchmark.harness import run_benchmark

    seen: list[tuple[int, int, str]] = []
    run_benchmark([_fake_trap()], {}, on_trap=lambda i, n, t: seen.append((i, n, t.id)))
    assert seen == [(1, 1, "fake_trap")]


def test_refresh_writes_baseline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "llm_baseline.json"
    out = llm_baseline.refresh(
        [_fake_trap()],
        _FakeClient("VERDICT: sound"),
        model="fake-model",
        captured_at="2026-07-21",
        path=path,
    )
    assert path.exists()
    assert out["fake_trap"]["verdict"] == "sound"
    assert llm_baseline.load_baseline(path) == out
