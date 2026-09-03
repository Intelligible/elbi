"""``elbi run``: trigger and watch runs on a running app (the operate surface).

The counterpart to ``sync``: where ``sync`` pushes declarative files, ``run`` triggers
the side-effecting operations (materialize assets, run a workflow or notebook, backfill)
over the app's HTTP API and reports the outcome with a nonzero exit on failure, so a CI
job can ``sync`` then ``run`` then assert. It is a thin client of the same endpoints the
MCP operate tools call; the logic lives once, in the app.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import typer

from .._console import arrow, fail, ok
from ..http import RUN_TIMEOUT, client_for

run_app = typer.Typer(help="Trigger runs on a running app (materialize, workflows).")

_TERMINAL = {"succeeded", "failed", "cancelled"}


def _client(url: str | None, token: str | None) -> httpx.Client:
    """An HTTP client for the app; runs can be slow, so the timeout is generous."""
    return client_for(url, token, timeout=RUN_TIMEOUT)


def _poll(client: httpx.Client, run_id: str) -> dict[str, Any]:
    """Poll a run to a terminal state and return its detail."""
    while True:
        detail: dict[str, Any] = client.get(f"/api/orchestration/runs/{run_id}").json()
        if detail["status"] in _TERMINAL:
            return detail
        time.sleep(0.5)


def _finish(detail: dict[str, Any]) -> None:
    """Report a finished run and exit nonzero if it failed."""
    status = detail["status"]
    if status == "succeeded":
        ok(f"run {detail['id']}: succeeded")
    else:
        fail(f"run {detail['id']}: {status}")
        raise typer.Exit(code=1)


@run_app.command("materialize")
def materialize(
    select: str = typer.Option("stale", help="'stale' or 'all'."),
    asset: list[str] = typer.Option(
        None, "--asset", help="Materialize a specific asset (repeatable)."
    ),
    downstream: bool = typer.Option(
        False, help="Also materialize the named assets' downstream."
    ),
    url: str = typer.Option(None, help="App base URL."),
    token: str = typer.Option(None, help="API key, if required."),
) -> None:
    """Materialize assets and wait for the run to finish."""
    body: dict[str, Any] = {"selection": select, "include_downstream": downstream}
    if asset:
        body["assets"] = list(asset)
    with _client(url, token) as client:
        started = client.post("/api/orchestration/materialize", json=body).json()
        detail = _poll(client, started["runId"])
        for step in detail["steps"]:
            arrow(f"{step['asset']:24} {step['state']}")
        _finish(detail)


@run_app.command("workflow")
def workflow(
    name: str = typer.Argument(..., help="Workflow name."),
    url: str = typer.Option(None, help="App base URL."),
    token: str = typer.Option(None, help="API key, if required."),
) -> None:
    """Run a workflow by name and wait for it to finish."""
    with _client(url, token) as client:
        match = next(
            (
                w
                for w in client.get("/api/orchestration/workflows").json()
                if w["name"] == name
            ),
            None,
        )
        if match is None:
            fail(f"no workflow named {name!r}")
            raise typer.Exit(code=1)
        run_id = client.post(f"/api/orchestration/workflows/{match['id']}/run").json()[
            "runId"
        ]
        detail = _poll(client, run_id)
        for step in detail.get("workflowSteps", []):
            arrow(f"{step['id']:24} {step['state']}")
        _finish(detail)


@run_app.command("backfill")
def backfill(
    asset: str = typer.Argument(..., help="Parameterized asset to backfill."),
    param: str = typer.Option(..., help="The partition parameter name."),
    values: str = typer.Option(..., help="Comma-separated partition values."),
    url: str = typer.Option(None, help="App base URL."),
    token: str = typer.Option(None, help="API key, if required."),
) -> None:
    """Backfill a parameterized asset across partition values, and wait."""
    partitions = [v.strip() for v in values.split(",") if v.strip()]
    body = {"asset": asset, "param": param, "values": partitions}
    with _client(url, token) as client:
        run_id = client.post("/api/orchestration/backfill", json=body).json()["runId"]
        detail = _poll(client, run_id)
        arrow(f"{len(detail['steps'])} partition(s)")
        _finish(detail)


@run_app.command("notebook")
def notebook(
    name: str = typer.Argument(..., help="Notebook name."),
    url: str = typer.Option(None, help="App base URL."),
    token: str = typer.Option(None, help="API key, if required."),
) -> None:
    """Run every cell of a notebook to completion and report the outcome."""
    with _client(url, token) as client:
        match = next(
            (n for n in client.get("/api/notebooks").json() if n["name"] == name),
            None,
        )
        if match is None:
            fail(f"no notebook named {name!r}")
            raise typer.Exit(code=1)
        statuses: dict[str, str] = {}
        try:
            with client.stream(
                "POST", f"/api/notebooks/{match['id']}/run", json={"run_all": True}
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    if event.get("event") == "status" and "cell" in event:
                        statuses[event["cell"]] = str(event.get("status"))
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout) as exc:
            # The run is the app's, not this stream's: the POST started it, and
            # losing the connection does not stop it. Retrying would start a second
            # run, so say what is known and where to look instead of guessing.
            fail(f"lost the connection while the notebook was running: {exc}")
            arrow(
                f"the run may still be going. Check the notebook {name!r} in the app, "
                "or re-run once it has finished."
            )
            raise typer.Exit(code=1) from exc
        failed = [cell for cell, status in statuses.items() if status != "ok"]
        for cell, status in statuses.items():
            arrow(f"{cell[:12]:14} {status}")
        if failed:
            fail(f"notebook {name!r}: {len(failed)} cell(s) failed")
            raise typer.Exit(code=1)
        ok(f"notebook {name!r}: {len(statuses)} cell(s) ran")


def _poll_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    """Poll a background job to a terminal state and return its record."""
    while True:
        detail: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if detail["state"] in {"succeeded", "failed", "cancelled"}:
            return detail
        time.sleep(0.5)


@run_app.command("train")
def train(
    name: str = typer.Argument(..., help="Model name (must have a retrain policy)."),
    url: str = typer.Option(None, help="App base URL."),
    token: str = typer.Option(None, help="API key, if required."),
) -> None:
    """Manually train a model from its retrain policy, waiting for the run.

    Reads the model's ``models/<name>.yaml`` policy (source, target, engine, budget) and
    submits a training job: a manual trigger of the same spec the policy fires on a
    schedule or data change.
    """
    with _client(url, token) as client:
        policy = client.get(f"/api/registry/models/{name}/retrain").json()
        if not policy.get("configured"):
            fail(f"no retrain policy for {name!r}; define models/{name}.yaml and sync")
            raise typer.Exit(code=1)
        source_kind = policy.get("sourceKind") or "dataset"
        payload: dict[str, Any] = {
            "name": name,
            source_kind: policy["dataset"],
            "target": policy["target"],
            "features": policy.get("features") or [],
            "task": policy.get("task") or "auto",
            "time_budget": policy.get("timeBudget") or 60.0,
        }
        if policy.get("engine"):
            payload["engine"] = policy["engine"]
        if policy.get("metric"):
            payload["metric"] = policy["metric"]
        job = client.post("/api/registry/train", json=payload).json()
        arrow(f"training {name!r} (job {job['id']})...")
        detail = _poll_job(client, job["id"])
        if detail["state"] == "succeeded":
            ok(f"trained {name!r}: {detail.get('result') or 'done'}")
        else:
            fail(f"train {name!r}: {detail['state']}, {detail.get('error')}")
            raise typer.Exit(code=1)


@run_app.command("status")
def status(
    url: str = typer.Option(None, help="App base URL."),
    token: str = typer.Option(None, help="API key, if required."),
) -> None:
    """List every asset's freshness and oracle verdict (read-only)."""
    with _client(url, token) as client:
        rows = client.get("/api/orchestration/status").json()
    if not rows:
        ok("no assets.")
        return
    for row in rows:
        verdict = f" [{row['verdict']}]" if row.get("verdict") else ""
        arrow(f"{row['asset']:24} {row['status']}{verdict}")
