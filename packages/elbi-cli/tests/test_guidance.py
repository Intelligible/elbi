"""Tests for the analysis-guidance instructions surfaced over MCP."""

from __future__ import annotations

from elbi_cli.guidance import ANALYSIS_GUIDANCE, compose_instructions


def test_default_guidance_is_question_agnostic_and_substantive() -> None:
    # It encodes general analytical discipline, not a per-question-type recipe.
    for phrase in (
        "careful data scientist",
        "confounders",
        "association from causation",
        "uncertainty",
        "author",
    ):
        assert phrase in ANALYSIS_GUIDANCE


def test_compose_without_project_context_returns_default() -> None:
    assert compose_instructions(None) == ANALYSIS_GUIDANCE


def test_compose_appends_project_context() -> None:
    out = compose_instructions("We model retail sales; revenue is in USD.")
    assert out.startswith(ANALYSIS_GUIDANCE)
    assert "Project context:" in out
    assert "retail sales" in out
