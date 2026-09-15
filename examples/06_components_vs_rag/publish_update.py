"""The simultaneous-update trigger: detention goes from $50/hour to $75/hour.

`run()` does two things from one call, so neither side gets a head start in a live
demo:

  1. RAG side: append a new document to the runtime `rag_docs/` directory. The
     original `base_tariff.md` (still saying $50/hour) is never touched -- see
     `rag/pipeline.py`'s module docstring for why that's the point, not an
     oversight.
  2. Elbi side: mark the current `detention_fee` row in the runtime
     `rate_schedule.csv` superseded (`valid_until` gets today's date) and append a
     new row at $75/hour with `valid_until` empty and `supersedes` pointing at the
     old row's component id. `derivations/freight_components.py` recomputes on its
     next call (its cache is keyed on this file's content hash), returning both --
     the new one as the live answer, the old one still there as an audit trail
     `agent/demo_agent.py` no longer searches but a client can still follow via the
     `supersedes` relation.

Idempotent: if detention is already at $75/hour, `run()` reports that and changes
nothing, so calling this twice (a double click on "Publish update" in the UI, or a
retried demo) can't corrupt the state.

Never touches the committed `fixtures/`; always operates on a `runtime/` working
copy (see `ensure_runtime` below), which `.gitignore` excludes for exactly this
reason.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ids import version_id

EXAMPLE_ROOT = Path(__file__).resolve().parent
FIXTURES = EXAMPLE_ROOT / "fixtures"
DEFAULT_RUNTIME = EXAMPLE_ROOT / "runtime"

_FIELDNAMES = [
    "key", "type", "item", "rate", "unit", "rate_display", "detail",
    "citation", "effective_date", "valid_until", "depends_on", "supersedes",
]  # fmt: skip

_NEW_RATE = 75
_NEW_RATE_DISPLAY = "$75/hour"
_NEW_CITATION = "ACME Freight Accessorial Tariff Amendment #1"


def ensure_runtime(runtime_dir: Path = DEFAULT_RUNTIME) -> Path:
    """Return `runtime_dir`, seeding it from the committed fixtures on first use.

    Safe to call every time: only copies what's missing, never overwrites a
    runtime file that already exists (so a demo's live edits survive a restart).
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "rag_docs").mkdir(exist_ok=True)

    csv_path = runtime_dir / "rate_schedule.csv"
    if not csv_path.exists():
        shutil.copy2(FIXTURES / "rate_schedule.csv", csv_path)

    for source in (FIXTURES / "rag_docs").glob("*.md"):
        target = runtime_dir / "rag_docs" / source.name
        if not target.exists():
            shutil.copy2(source, target)

    return runtime_dir


@dataclass
class UpdateResult:
    applied: bool
    detail: str
    old_component_id: str | None = None
    new_component_id: str | None = None


def run(
    runtime_dir: Path = DEFAULT_RUNTIME, *, today: date | None = None
) -> UpdateResult:
    """Apply the detention-fee update. Idempotent; see the module docstring."""
    runtime_dir = ensure_runtime(runtime_dir)
    today = today or date.today()
    effective_date = today.isoformat()

    csv_path = runtime_dir / "rate_schedule.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    current = [
        row for row in rows if row["key"] == "detention_fee" and not row["valid_until"]
    ]
    if not current:
        return UpdateResult(applied=False, detail="detention_fee has no current row")
    if len(current) > 1:
        raise RuntimeError("more than one current detention_fee row; state is corrupt")

    old_row = current[0]
    if float(old_row["rate"]) == _NEW_RATE:
        return UpdateResult(
            applied=False,
            detail="already at $75/hour; run() is a no-op",
            old_component_id=None,
            new_component_id=version_id("detention_fee", old_row["effective_date"]),
        )

    old_row["valid_until"] = effective_date
    old_id = version_id("detention_fee", old_row["effective_date"])
    new_id = version_id("detention_fee", effective_date)

    new_row = {
        **old_row,
        "rate": str(_NEW_RATE),
        "rate_display": _NEW_RATE_DISPLAY,
        "citation": f"{_NEW_CITATION}, effective {effective_date}",
        "effective_date": effective_date,
        "valid_until": "",
        "supersedes": old_id,
    }
    rows.append(new_row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    _publish_rag_amendment(runtime_dir, effective_date=effective_date)

    return UpdateResult(
        applied=True,
        detail=(
            f"detention_fee: $50/hour -> {_NEW_RATE_DISPLAY}, "
            f"effective {effective_date}"
        ),
        old_component_id=old_id,
        new_component_id=new_id,
    )


def _publish_rag_amendment(runtime_dir: Path, *, effective_date: str) -> None:
    """Add a new document to the RAG corpus. Never edits `base_tariff.md`."""
    amendment_path = runtime_dir / "rag_docs" / "amendment_1.md"
    if amendment_path.exists():
        return  # idempotent, matching the CSV side

    amendment_path.write_text(
        "# ACME Freight Accessorial Tariff -- Amendment #1 "
        f"(effective {effective_date})\n\n"
        "## 4.2 Detention\n\n"
        f"Detention (after 2 free hours) costs {_NEW_RATE_DISPLAY}. This supersedes "
        "the $50/hour rate in the base tariff, effective the date above.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME,
        help="Working copy to update (seeded from fixtures/ on first use).",
    )
    args = parser.parse_args()
    result = run(args.runtime_dir)
    print(("applied: " if result.applied else "no-op: ") + result.detail)


if __name__ == "__main__":
    main()
