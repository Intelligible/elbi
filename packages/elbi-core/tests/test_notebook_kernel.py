"""Tests for the notebook kernel: rich output, persistence, errors, interrupt.

These run a real worker subprocess but need no third-party packages, so they are fast.
They exercise the behaviors a notebook depends on: a cell's last expression comes back
as
a result, the namespace survives across cells, an error is reported without killing the
kernel, and a runaway cell is interrupted while its namespace lives on.
"""

from __future__ import annotations

from typing import Any

import pytest

from elbi_core.notebook import SubprocessKernel


def _collect(
    kernel: SubprocessKernel, code: str, count: int = 1
) -> tuple[str, list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    status = kernel.execute(code, count, outputs.append)
    return status, outputs


def _results(outputs: list[dict[str, Any]]) -> list[str]:
    """The text/plain of each execute_result output."""
    return [o["data"]["text/plain"] for o in outputs if o["type"] == "execute_result"]


def test_output_written_to_the_raw_fd_is_captured() -> None:
    # Output a C extension or a pre-bound logging handler writes to the file descriptor
    # by number (not via sys.stdout) is still captured and streamed to the frontend.
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(kernel, "import os\nos.write(2, b'via-raw-fd\\n')")
        assert status == "ok"
        streamed = "".join(o.get("text", "") for o in outputs if o["type"] == "stream")
        assert "via-raw-fd" in streamed
    finally:
        kernel.close()


def test_kernel_seeds_datasets_with_typed_warehouse_values() -> None:
    # A typed warehouse column yields dates and decimals; the kernel must still start
    # (JSON has no native form for them) and hand the cell their natural values.
    import datetime as dt
    from decimal import Decimal

    kernel = SubprocessKernel(
        cell_timeout=30,
        data={"t": [{"day": dt.date(2025, 1, 2), "amt": Decimal("3.50")}]},
    )
    try:
        status, outputs = _collect(kernel, "r = data['t'][0]\n(r['day'], r['amt'])")
        assert status == "ok"
        assert _results(outputs) == ["('2025-01-02', 3.5)"]
    finally:
        kernel.close()


def test_last_expression_is_a_result_and_namespace_persists() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(kernel, "x = 41\nx + 1")
        assert status == "ok"
        assert _results(outputs) == ["42"]
        # A later cell sees what the first defined.
        status, outputs = _collect(kernel, "x * 2", 2)
        assert _results(outputs) == ["82"]
    finally:
        kernel.close()


def test_stdout_is_streamed() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _, outputs = _collect(kernel, "print('hello')")
        streams = [o for o in outputs if o["type"] == "stream"]
        assert streams and "hello" in streams[0]["text"]
    finally:
        kernel.close()


def test_error_is_reported_and_kernel_survives() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(kernel, "y = 7\n1 / 0")
        assert status == "error"
        assert any(
            o["type"] == "error" and o["ename"] == "ZeroDivisionError" for o in outputs
        )
        # The kernel is still alive and the pre-error binding persists.
        status, outputs = _collect(kernel, "y", 2)
        assert status == "ok"
        assert _results(outputs) == ["7"]
    finally:
        kernel.close()


def test_display_protocol_emits_a_bundle() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _, outputs = _collect(kernel, "display({'a': 1})")
        displays = [o for o in outputs if o["type"] == "display_data"]
        assert displays and "text/plain" in displays[0]["data"]
    finally:
        kernel.close()


def test_timeout_interrupts_but_keeps_namespace() -> None:
    kernel = SubprocessKernel(cell_timeout=2)
    try:
        status, _ = _collect(kernel, "kept = 5\nwhile True:\n    pass")
        assert status == "interrupted"
        assert kernel.alive
        # The binding made before the infinite loop is still there.
        status, outputs = _collect(kernel, "kept", 2)
        assert status == "ok"
        assert _results(outputs) == ["5"]
    finally:
        kernel.close()


def test_register_model_refuses_when_tracking_is_unconfigured() -> None:
    # MLflow's fallback writes to a directory inside the sandbox and reports success, so
    # the reflex must name the missing setting rather than lose the run quietly.
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(kernel, "register_model('churn')")
        assert status == "error"
        errors = [o for o in outputs if o["type"] == "error"]
        assert errors and "NOTEBOOK_MLFLOW_TRACKING_URI" in errors[0]["evalue"]
    finally:
        kernel.close()


def test_a_tracking_uri_leaves_mlflows_own_names_alone() -> None:
    # With somewhere real to log, the sentinel must not shadow the library's own call.
    kernel = SubprocessKernel(
        cell_timeout=30, env={"MLFLOW_TRACKING_URI": "http://tracking.example/mlflow"}
    )
    try:
        status, outputs = _collect(
            kernel,
            "import os\n"
            "print(os.environ['MLFLOW_TRACKING_URI'])\n"
            "print('register_model' in dir(__builtins__))",
        )
        assert status == "ok"
        printed = "".join(o.get("text", "") for o in outputs)
        assert "http://tracking.example/mlflow" in printed
        assert "False" in printed
    finally:
        kernel.close()


def test_syntax_error_is_reported() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(kernel, "def broken(")
        assert status == "error"
        assert any(o["type"] == "error" for o in outputs)
    finally:
        kernel.close()


def _streams(outputs: list[dict[str, Any]]) -> str:
    """Concatenated text of every stream output."""
    return "".join(o.get("text", "") for o in outputs if o["type"] == "stream")


def test_shell_escape_and_capture() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _, outputs = _collect(kernel, "!echo hello_shell")
        assert "hello_shell" in _streams(outputs)
        # `x = !cmd` captures the command's lines into a list.
        _, outputs = _collect(kernel, "lines = !printf 'a\\nb\\n'\nlines", 2)
        assert _results(outputs) == ["['a', 'b']"]
    finally:
        kernel.close()


def test_line_and_cell_magics() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _, outputs = _collect(kernel, "%time sum(range(100))")
        assert "Wall time" in _streams(outputs)
        assert _results(outputs) == ["4950"]  # %time still shows the value

        _, outputs = _collect(kernel, "%%bash\necho from_bash", 2)
        assert "from_bash" in _streams(outputs)

        # A cell magic can emit rich output.
        _, outputs = _collect(kernel, "%%html\n<b>hi</b>", 3)
        html = [o for o in outputs if o["type"] == "display_data"]
        assert html and html[0]["data"]["text/html"] == "<b>hi</b>"

        # An unknown magic is a clean error, not a crash.
        status, outputs = _collect(kernel, "%frobnicate", 4)
        assert status == "error"
        assert any(o["type"] == "error" and o["ename"] == "UsageError" for o in outputs)
    finally:
        kernel.close()


def test_completion_from_the_namespace() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _collect(kernel, "import math\nmyframe = 1\nnums = [1, 2]")
        assert "myframe" in kernel.complete("myfr", 4)["matches"]
        assert "sqrt" in kernel.complete("math.sq", 7)["matches"]
        assert "append" in kernel.complete("nums.app", 8)["matches"]
        # cursor_start marks where the replaced token begins.
        reply = kernel.complete("myfr", 4)
        assert reply["cursor_start"] == 0 and reply["cursor_end"] == 4
    finally:
        kernel.close()


def test_inspection_of_a_name() -> None:
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _collect(kernel, "def greet(name):\n    'Say hi.'\n    return 'hi ' + name")
        reply = kernel.inspect("greet", 3)
        assert reply["found"] is True
        text = reply["data"]["text/plain"]
        assert "greet(name)" in text and "Say hi." in text
        # A cursor not on a resolvable name finds nothing.
        assert kernel.inspect("nosuchname", 4)["found"] is False
    finally:
        kernel.close()


def test_input_requests_and_receives_a_value() -> None:
    import threading
    import time

    kernel = SubprocessKernel(cell_timeout=30)
    outputs: list[dict[str, Any]] = []

    def answer() -> None:
        for _ in range(300):
            if any(o.get("type") == "input_request" for o in outputs):
                break
            time.sleep(0.02)
        kernel.send_input("Ada")

    try:
        feeder = threading.Thread(target=answer)
        feeder.start()
        status = kernel.execute(
            "name = input('your name? ')\nprint('hi', name)", 1, outputs.append
        )
        feeder.join()
        assert status == "ok"
        requests = [o for o in outputs if o["type"] == "input_request"]
        assert requests and requests[0]["prompt"] == "your name? "
        assert "hi Ada" in _streams(outputs)
    finally:
        kernel.close()


def test_ipywidgets_comm_round_trip() -> None:
    """A widget syncs state both ways over the kernel's comm channel.

    Proves the ipywidgets comm relay: constructing a widget opens a comm carrying its
    state, its display carries the widget-view model id, a frontend update reaches the
    Python model, and a Python change emits an update back.
    """
    import time

    pytest.importorskip("ipywidgets")
    comms: list[dict[str, Any]] = []
    kernel = SubprocessKernel(cell_timeout=60)
    kernel.set_comm_listener(comms.append)
    try:
        _, outputs = _collect(
            kernel, "import ipywidgets as w\nslider = w.IntSlider(value=3)\nslider"
        )
        views = [
            o
            for o in outputs
            if "application/vnd.jupyter.widget-view+json" in o.get("data", {})
        ]
        assert views, "the widget's display should carry a widget-view mimetype"
        model_id = views[0]["data"]["application/vnd.jupyter.widget-view+json"][
            "model_id"
        ]
        opened = next(
            m
            for m in comms
            if m["type"] == "comm_open" and m["content"]["comm_id"] == model_id
        )
        assert opened["content"]["data"]["state"]["value"] == 3

        # Frontend -> kernel: an update message sets the trait on the Python model.
        kernel.send_comm(
            {
                "op": "comm_msg",
                "content": {
                    "comm_id": model_id,
                    "data": {"method": "update", "state": {"value": 7}},
                },
                "buffers": [],
            }
        )
        time.sleep(0.4)
        _, outputs = _collect(kernel, "slider.value", 2)
        assert _results(outputs) == ["7"]

        # Kernel -> frontend: a Python change emits a comm_msg update with the value.
        comms.clear()
        _collect(kernel, "slider.value = 42", 3)
        time.sleep(0.4)
        values = [
            m["content"]["data"].get("state", {}).get("value")
            for m in comms
            if m["type"] == "comm_msg" and m["content"]["comm_id"] == model_id
        ]
        assert 42 in values
    finally:
        kernel.close()


def test_a_query_result_carrying_typed_values_reaches_the_cell() -> None:
    """A date, timestamp or decimal in a result must not wedge the kernel.

    These are the ordinary types of a warehouse column, and the reply crosses to the
    worker as JSON, which has no native form for any of them. The worker blocks reading
    that reply, so a value the encoder cannot write is not a failed query -- it is a
    cell that waits for an answer that never arrives, until the deadline kills it with
    a bare KeyboardInterrupt naming nothing.

    Asserted on the values the cell actually receives, not merely that it finished,
    because answering with an error would also unblock it.
    """
    import datetime as dt
    from decimal import Decimal

    def resolve(sql: str, table: str, limit: int) -> dict[str, Any]:
        return {
            "columns": ["day", "at", "amount"],
            "rows": [
                {
                    "day": dt.date(2024, 3, 1),
                    "at": dt.datetime(2024, 3, 1, 12, 30, 5),
                    "amount": Decimal("19.99"),
                }
            ],
            "truncated": False,
        }

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["t"])
    kernel.set_query_resolver(resolve)
    try:
        status, outputs = _collect(
            kernel, "[sql('select 1').rows[0][c] for c in ('day','at','amount')]"
        )
        assert status == "ok", outputs
        assert _results(outputs) == ["['2024-03-01', '2024-03-01T12:30:05', 19.99]"]
    finally:
        kernel.close()


def test_a_query_result_that_cannot_encode_answers_rather_than_hanging() -> None:
    """The backstop: an unencodable value returns an error, and the cell continues.

    Without it the worker waits on a reply that was never written, which presents as a
    hang rather than as the failure it is.
    """

    def resolve(sql: str, table: str, limit: int) -> dict[str, Any]:
        return {"columns": ["x"], "rows": [{"x": object()}], "truncated": False}

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["t"])
    kernel.set_query_resolver(resolve)
    try:
        status, outputs = _collect(kernel, "sql('select 1')")
        assert status == "error", outputs
        assert "cannot be sent" in "".join(str(o.get("evalue", "")) for o in outputs), (
            outputs
        )
    finally:
        kernel.close()


def test_lazy_data_fetches_on_access_and_not_before() -> None:
    # The point of the lazy path: starting a kernel costs nothing, and a dataset only
    # crosses into it when a cell actually names it.
    requests: list[dict[str, Any]] = []

    def resolve(sql: str, table: str, limit: int) -> dict[str, Any]:
        requests.append({"sql": sql, "table": table, "limit": limit})
        return {"columns": ["n"], "rows": [{"n": 1}, {"n": 2}], "truncated": False}

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders", "unused"])
    kernel.set_query_resolver(resolve)
    try:
        # Naming the datasets without reading them must not fetch anything.
        status, outputs = _collect(kernel, "sorted(data)")
        assert status == "ok"
        assert _results(outputs) == ["['orders', 'unused']"]
        assert requests == [], "listing datasets must not fetch rows"

        # Reading one fetches exactly that one.
        status, outputs = _collect(kernel, "len(data['orders'])", 2)
        assert status == "ok"
        assert _results(outputs) == ["2"]
        assert len(requests) == 1
        # A second read is cached, not re-fetched.
        status, _ = _collect(kernel, "data['orders']", 3)
        assert status == "ok"
        assert len(requests) == 1, "a second access must come from cache"
        # `unused` was never touched, so it was never transferred.
        assert all(r["table"] != "unused" for r in requests)
    finally:
        kernel.close()


def test_sql_asks_for_the_whole_result_and_reports_truncation() -> None:
    """A cell's ``sql()`` gets the whole result; the *view* is what is bounded.

    Capping the data is the trap: ``select * from t`` returning a thousand rows of a
    21,613-row table, with only an attribute to say so, makes a model fitted to that
    look fine. Databricks caps ``display`` and leaves the DataFrame whole.
    """

    def resolve(sql: str, table: str, limit: int) -> dict[str, Any]:
        # Echo the bound the kernel asked for, so the contract is asserted directly.
        return {
            "columns": ["asked_for", "n"],
            "rows": [{"asked_for": limit, "n": 3}],
            "truncated": False,
        }

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders"])
    kernel.set_query_resolver(resolve)
    try:
        status, outputs = _collect(
            kernel, "sql('select 1')[0]['asked_for'] > 1_000_000"
        )
        assert status == "ok"
        assert _results(outputs) == ["True"], "the default asked for a small bound"
        # An explicit limit is still a deliberate, honoured narrowing.
        status, outputs = _collect(
            kernel, "sql('select 1', limit=5)[0]['asked_for']", 2
        )
        assert status == "ok"
        assert _results(outputs) == ["5"]
    finally:
        kernel.close()


def test_query_error_surfaces_as_a_cell_error_not_a_hang() -> None:
    # A resolver reporting a bad query must produce a normal cell error. A cell parked
    # in sql() reads until it gets a reply, so failing to answer would hang it.
    def resolve(sql: str, table: str, limit: int) -> dict[str, Any]:
        return {"error": 'Referenced column "nope" not found'}

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders"])
    kernel.set_query_resolver(resolve)
    try:
        status, outputs = _collect(kernel, "sql('select nope from orders')")
        assert status == "error"
        errors = [o for o in outputs if o["type"] == "error"]
        assert errors and "not found" in str(errors[0])
    finally:
        kernel.close()


def test_missing_resolver_errors_rather_than_hanging() -> None:
    # A kernel started without a warehouse must tell the cell so, promptly.
    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders"])
    try:
        status, outputs = _collect(kernel, "sql('select 1')")
        assert status == "error"
        errors = [o for o in outputs if o["type"] == "error"]
        assert errors and "no query resolver" in str(errors[0])
    finally:
        kernel.close()


def test_runaway_output_is_truncated_and_the_cell_still_finishes() -> None:
    # A cell's outputs are held in memory, stored as JSON and sent to a browser, so a
    # print loop is a cost to the host, not to the sandbox that wrote it. The budget is
    # enforced where the host reads, and the cell is left to finish: truncating output
    # is not a reason to fail a run.
    kernel = SubprocessKernel(cell_timeout=30, max_output_bytes=64 * 1024)
    try:
        status, outputs = _collect(
            kernel, "for _ in range(200):\n    print('x' * 4096)\n'finished'"
        )
        assert status == "ok"
        streamed = "".join(o.get("text", "") for o in outputs if o["type"] == "stream")
        # Bounded, with a little headroom for the message that says so.
        assert len(streamed) < 96 * 1024
        assert "Output truncated at 65536 B" in streamed
        # What the cell was for survives the printing that crowded it out.
        assert _results(outputs) == ["'finished'"]
    finally:
        kernel.close()


def test_ordinary_output_is_not_truncated() -> None:
    kernel = SubprocessKernel(cell_timeout=30, max_output_bytes=64 * 1024)
    try:
        status, outputs = _collect(kernel, "print('y' * 1000)\n'done'")
        assert status == "ok"
        streamed = "".join(o.get("text", "") for o in outputs if o["type"] == "stream")
        assert streamed == "y" * 1000 + "\n"
        assert _results(outputs) == ["'done'"]
    finally:
        kernel.close()


def test_a_traceback_survives_truncation() -> None:
    # The diagnosis is the point of a failed cell; losing it to preceding output would
    # leave a user with a red cell and no reason for it.
    kernel = SubprocessKernel(cell_timeout=30, max_output_bytes=64 * 1024)
    try:
        status, outputs = _collect(
            kernel,
            "for _ in range(200):\n    print('x' * 4096)\n"
            "raise ValueError('the cause')",
        )
        assert status == "error"
        errors = [o for o in outputs if o["type"] == "error"]
        assert errors and "the cause" in "".join(errors[0]["traceback"])
    finally:
        kernel.close()


def test_a_cell_reports_which_side_of_the_data_boundary_it_used() -> None:
    """The crossing has to be visible, which means the kernel has to report it.

    A lazy handle whose materialisation is invisible only moves the memory failure
    later. ``sql()`` leaves the rows in the warehouse and ``data['x']`` brings them
    here; a cell that does both has materialised, since the cheaper half does not undo
    the expensive one.
    """
    rows = [{"n": 1}, {"n": 2}]

    def resolver(sql: str, table: str, limit: int) -> dict[str, Any]:
        return {"columns": ["n"], "rows": rows, "truncated": False}

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders", "customers"])
    kernel.set_query_resolver(resolver)
    try:
        # A cell that touches no data says nothing.
        _collect(kernel, "1 + 1", 1)
        assert kernel.last_data_mode == ""

        _collect(kernel, "sql('select 1')", 2)
        assert kernel.last_data_mode == "pushed_down"

        _collect(kernel, "data['orders']", 3)
        assert kernel.last_data_mode == "materialised"

        # Reported per cell, not accumulated across the session.
        _collect(kernel, "sql('select 1')", 4)
        assert kernel.last_data_mode == "pushed_down"

        # Both in one cell: materialising is the fact worth surfacing. A table not yet
        # cached, since a cached read crosses nothing: see the next test.
        _collect(kernel, "sql('select 1'); data['customers']", 5)
        assert kernel.last_data_mode == "materialised"
    finally:
        kernel.close()


def test_a_cached_table_is_not_a_second_crossing() -> None:
    calls: list[str] = []

    def resolver(sql: str, table: str, limit: int) -> dict[str, Any]:
        calls.append(table or sql)
        return {"columns": ["n"], "rows": [{"n": 1}], "truncated": False}

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders"])
    kernel.set_query_resolver(resolver)
    try:
        _collect(kernel, "data['orders']", 1)
        assert kernel.last_data_mode == "materialised"
        # The second read is served from the kernel's cache, so nothing crosses and the
        # cell is not labelled as though something had.
        _collect(kernel, "data['orders']", 2)
        assert kernel.last_data_mode == ""
        assert calls == ["orders"]
    finally:
        kernel.close()


def test_a_cell_records_what_it_pushed_to_the_warehouse() -> None:
    """The pushdown claim has to be checkable, which means recording the query.

    Same fields Databricks' query history keeps (statement, duration, rows produced)
    minus the ones a single-node engine cannot honestly report.
    """
    import time as clock

    def resolver(sql: str, table: str, limit: int) -> dict[str, Any]:
        clock.sleep(0.01)
        if "boom" in sql:
            return {"error": "Binder Error: no such column"}
        return {"columns": ["n"], "rows": [{"n": 1}], "truncated": True}

    kernel = SubprocessKernel(cell_timeout=30, dataset_names=["orders"])
    kernel.set_query_resolver(resolver)
    try:
        _collect(kernel, "sql('select 1'); sql('select 2')", 1)
        assert [q["sql"] for q in kernel.last_queries] == ["select 1", "select 2"]
        assert all(q["duration_ms"] > 0 for q in kernel.last_queries)
        assert all(q["rows"] == 1 and q["truncated"] for q in kernel.last_queries)

        # Per cell, not accumulated across the session.
        _collect(kernel, "1 + 1", 2)
        assert kernel.last_queries == []

        # A whole-table read is recorded as what it is, since the kernel never sees SQL
        # for it -- the host resolves the name.
        _collect(kernel, "data['orders']", 3)
        assert kernel.last_queries[0]["sql"] == "read table orders"

        # A failed query is exactly the one someone needs to see, so it is kept with its
        # error and its duration rather than dropped.
        _collect(kernel, "sql('select boom')", 4)
        assert kernel.last_queries[0]["error"] == "Binder Error: no such column"
        assert kernel.last_queries[0]["duration_ms"] > 0
    finally:
        kernel.close()


def test_defining_a_derivation_shows_what_it_returns() -> None:
    """A decorated `def` has no value to echo, so the kernel runs it and shows one.

    Someone editing a derivation in a notebook is editing it to see what changed; a
    cell that ran clean and displayed nothing is the same as one that did not run.
    """
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(
            kernel,
            "from elbi_core import Artifact, Context, derivation\n"
            "@derivation()\n"
            "def totals(ctx):\n"
            "    return Artifact.table([{'n': 1}])",
        )

        assert status == "ok"
        html = [
            o["data"].get("text/html", "")
            for o in outputs
            if o["type"] == "execute_result"
        ]
        assert any("<td>1</td>" in rendering for rendering in html)
    finally:
        kernel.close()


def test_a_derivation_reading_an_upstream_runs_the_upstream_first() -> None:
    """The same resolution the runner does, so an edit upstream shows up downstream."""
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        _collect(
            kernel,
            "from elbi_core import Artifact, Context, derivation\n"
            "@derivation()\n"
            "def base(ctx):\n"
            "    return Artifact.table([{'n': 2}])",
        )
        status, outputs = _collect(
            kernel,
            "@derivation(inputs={'rows': base})\n"
            "def doubled(ctx):\n"
            "    rows = ctx.input('rows').value\n"
            "    return Artifact.table([{'n': r['n'] * 2} for r in rows])",
            2,
        )

        assert status == "ok"
        html = [
            o["data"].get("text/html", "")
            for o in outputs
            if o["type"] == "execute_result"
        ]
        assert any("<td>4</td>" in rendering for rendering in html)
    finally:
        kernel.close()


def test_a_derivation_taking_parameters_is_not_run_on_a_guess() -> None:
    """No values for them here, and inventing some would show a fabricated result."""
    kernel = SubprocessKernel(cell_timeout=30)
    try:
        status, outputs = _collect(
            kernel,
            "from elbi_core import Artifact, Context, derivation\n"
            "from elbi_core.param import integer\n"
            "@derivation(params={'n': integer()})\n"
            "def scaled(ctx):\n"
            "    return Artifact.table([{'n': ctx.param('n')}])",
        )

        assert status == "ok"
        assert _results(outputs) == []
    finally:
        kernel.close()
