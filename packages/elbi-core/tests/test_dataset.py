"""Tests for the Dataset reference."""

from __future__ import annotations

import pytest

from elbi_core import Dataset


def test_dataset_holds_name() -> None:
    assert Dataset("kc_house").name == "kc_house"


def test_dataset_is_frozen_and_hashable() -> None:
    assert Dataset("a") == Dataset("a")
    assert len({Dataset("a"), Dataset("a"), Dataset("b")}) == 2


def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Dataset("")
