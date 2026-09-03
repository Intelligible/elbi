"""Reading a CSV the way it was actually written, not the way we hope it was.

pyarrow's CSV reader assumes UTF-8 and a comma. Real files -- a spreadsheet export
especially -- are often something else: a Windows-1252 or Latin-1 encoding, a UTF-8 byte
order mark, a semicolon separator (the European default). Read with the wrong assumption
they fail outright or, worse, parse into one wide column of garbage.

So a sample of the file is inspected first, the encoding decided and the separator
sniffed, and the read options built from what was found. Both the file connector (at
sync) and the upload's validation use this, so a file that validates is one the sync can
read -- Metabase does the same detection at upload time and this closes the gap to it.

Encoding is decided by trying, not by a statistical detector: a small sample makes those
guess exotic encodings a real CSV never uses. Almost every file is UTF-8; the ones that
are not are almost always a Windows/Excel export, which is Windows-1252. So a strict
UTF-8 decode decides UTF-8, and anything else falls back to ``cp1252`` -- which decodes
any byte sequence, so it never fails and covers Latin-1 too. UTF-16, which would break
this, is caught by its byte order mark, the only form it takes for a CSV in practice.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.csv as pacsv

#: Bytes sampled to detect the dialect. Enough to be confident about the encoding and to
#: see several rows for the separator, small enough not to pull a multi-gigabyte object
#: over the wire twice.
_SAMPLE_BYTES = 64 * 1024

#: Separators worth considering. Comma and semicolon are the common spreadsheet exports;
#: tab and pipe cover the rest. The order is the preference when the sniffer is unsure.
_CANDIDATE_DELIMITERS = ",;\t|"


def detect_encoding(sample: bytes) -> str:
    """The text encoding of ``sample``, as a name pyarrow accepts.

    A byte order mark decides outright: ``utf-8-sig`` so the mark is stripped rather
    than smuggled into the first column's name, or ``utf-16`` for its marks. Then
    a strict UTF-8 decode decides UTF-8, and everything else is ``cp1252``.
    """
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        sample.decode("utf-8", errors="strict")
    except UnicodeDecodeError as err:
        # A multi-byte character split by the sample boundary is not a sign the file is
        # not UTF-8, only that the sample ended mid-character -- but that only happens
        # when the sample was cut off (a full 64 KiB read). If it is the whole file, a
        # bad byte at the end is a real one, so this is cp1252.
        truncated = len(sample) >= _SAMPLE_BYTES and err.start >= len(sample) - 3
        return "utf-8" if truncated else "cp1252"
    return "utf-8"


def detect_delimiter(sample: bytes, encoding: str) -> str:
    """The field separator in ``sample``, or a comma when it cannot be told.

    The sniffer decides from the decoded text; a decode error means the encoding guess
    was wrong for these bytes, in which case the separator cannot be trusted either and
    the comma default stands.
    """
    try:
        text = sample.decode(encoding, errors="strict")
    except (UnicodeDecodeError, LookupError):
        return ","
    try:
        return csv.Sniffer().sniff(text, delimiters=_CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        return ","


def read_options(sample: bytes) -> tuple[pacsv.ReadOptions, pacsv.ParseOptions]:
    """The pyarrow read and parse options matching the dialect of ``sample``."""
    import pyarrow.csv as pacsv

    encoding = detect_encoding(sample)
    delimiter = detect_delimiter(sample, encoding)
    return (
        pacsv.ReadOptions(encoding=encoding),
        pacsv.ParseOptions(delimiter=delimiter),
    )


def _options_for(source: Any) -> tuple[pacsv.ReadOptions, pacsv.ParseOptions]:
    """Sample ``source``, decide the dialect, and rewind it for the read."""
    sample = source.read(_SAMPLE_BYTES)
    source.seek(0)
    return read_options(sample)


def read_table(source: Any) -> pa.Table:
    """Read a whole CSV from a seekable binary ``source``, in its own dialect."""
    import pyarrow.csv as pacsv

    read_opts, parse_opts = _options_for(source)
    return pacsv.read_csv(source, read_options=read_opts, parse_options=parse_opts)


def open_reader(source: Any) -> pacsv.CSVStreamingReader:
    """A streaming CSV reader over a seekable binary ``source``, in its own dialect.

    For validating an upload: reading one batch confirms the header and first rows parse
    without pulling the whole file.
    """
    import pyarrow.csv as pacsv

    read_opts, parse_opts = _options_for(source)
    return pacsv.open_csv(source, read_options=read_opts, parse_options=parse_opts)
