"""Reading a CSV in the encoding and separator it was written in.

pyarrow assumes UTF-8 and a comma; a spreadsheet export is often neither. Detecting the
dialect first is what turns a Windows-1252, semicolon-separated file into columns rather
than one column of garbage -- the parity gap to Metabase, which detects it at upload.
"""

from __future__ import annotations

import io

import pyarrow.csv as pacsv
import pytest

from elbi.warehouse import csv_dialect


@pytest.mark.parametrize(
    ("label", "data", "encoding", "delimiter"),
    [
        ("utf-8 comma", "name,city\nJosé,São Paulo\n".encode(), "utf-8", ","),
        ("semicolon utf-8", "name;city\nJosé;Paris\n".encode(), "utf-8", ";"),
        ("cp1252 comma", "name,city\nJosé,Málaga\n".encode("cp1252"), "cp1252", ","),
        ("tab utf-8", b"name\tcity\nAda\tLondon\n", "utf-8", "\t"),
        (
            "utf-8 BOM",
            b"\xef\xbb\xbf" + b"name,city\nAda,London\n",
            "utf-8-sig",
            ",",
        ),
        (
            "semicolon cp1252",
            "name;city\nJosé;Köln\n".encode("cp1252"),
            "cp1252",
            ";",
        ),
    ],
)
def test_the_dialect_is_detected(
    label: str, data: bytes, encoding: str, delimiter: str
) -> None:
    assert csv_dialect.detect_encoding(data) == encoding
    assert csv_dialect.detect_delimiter(data, encoding) == delimiter


@pytest.mark.parametrize(
    "data",
    [
        "name,city\nJosé,São Paulo\n".encode(),
        "name;city\nJosé;Paris\n".encode(),
        "name,city\nJosé,Málaga\n".encode("cp1252"),
        "name;city\nJosé;Köln\n".encode("cp1252"),  # the case a naive read mangles
        b"\xef\xbb\xbf" + b"name,city\nAda,London\n",
    ],
)
def test_a_detected_read_yields_the_right_columns_and_values(data: bytes) -> None:
    """The end the detection exists for: real columns, real accented values, whatever
    the encoding and separator."""
    table = csv_dialect.read_table(io.BytesIO(data))
    assert table.column_names == ["name", "city"]
    assert table.num_rows == 1
    # the accented value survived the decode rather than becoming a replacement char
    assert "�" not in "".join(str(v) for v in table.to_pylist()[0].values())


def test_a_multibyte_char_split_by_the_sample_boundary_stays_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated sample that ends mid-character must not be misread as cp1252."""
    monkeypatch.setattr(csv_dialect, "_SAMPLE_BYTES", 12)
    # 'é' is two UTF-8 bytes; place it so the 12-byte (== the sample size) read cuts it.
    data = "abcdefghij,é\n".encode()
    assert csv_dialect.detect_encoding(data[:12]) == "utf-8"


def test_a_whole_small_file_ending_in_a_high_byte_is_cp1252() -> None:
    """The boundary allowance is only for a cut-off sample. When the sample is the whole
    file -- shorter than the read size -- a high byte at the end is a real one, not a
    truncated UTF-8 character, so it is cp1252 rather than a wrong UTF-8 guess."""
    assert csv_dialect.detect_encoding("café".encode("cp1252")) == "cp1252"


def test_undetectable_bytes_fall_back_rather_than_refuse() -> None:
    """cp1252 decodes any byte, so a genuinely odd file still reads -- better than a
    refusal a person cannot act on."""
    weird = b"name,city\n\x81\x9d,\x8f\n"  # bytes with no cp1252 meaning
    options = csv_dialect.read_options(weird)
    assert isinstance(options[0], pacsv.ReadOptions)
