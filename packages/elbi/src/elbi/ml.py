"""Model training, registry, and serving wired into the app (the ``ml`` extra).

One :class:`ModelService` per app instance adapts :mod:`elbi.ml` to the
app's three surfaces: the chat tool loop (rendered-string capabilities on the
:class:`~elbi_agent.Workspace`), the JSON API (``/api/registry/models``),
and the MLflow-protocol scoring route (``/api/serving/{name}/invocations``). The
tracking store resolves exactly like the certified-run export
(``MLFLOW_TRACKING_URI`` over the ``mlflow_tracking_uri`` setting), falling back
to a SQLite store under the
project's cache directory, so a team pointing the app at their MLflow server gets
their runs, registry, and models there, and everyone else gets a local one that
works with no configuration.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from elbi_core.errors import ModelError

from .db import Store
from .tracking import tracking_uri as configured_tracking_uri

logger = logging.getLogger("elbi")

#: Registry names must be safe in URIs, filters, and file paths.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_MIN_BUDGET = 5.0
#: An inline chat tool call blocks the turn, so its budget stays within the app's
#: run_code/derive wall-clock allowance; anything longer becomes a background job.
INLINE_TRAIN_BUDGET = 240.0
#: Background training jobs run for real-world durations (a day, not minutes).
_MAX_JOB_BUDGET = 86400.0

Rows = list[dict[str, str]]

#: Engineered rows echoed back by raw scoring, so the caller can see what the
#: feature derivation produced without re-deriving it.
_ENGINEERED_SAMPLE = 20

#: The environment variables MLflow's set_tracking_uri / set_registry_uri write; the
#: embedded server sets them per request, so the mount snapshots and restores them.
_MLFLOW_URI_ENV_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI")

#: An HTTP URL for this deployment's ``/mlflow`` mount, reachable *from a kernel*. Its
#: own setting because the app may hold a database URI, while a sandbox must be handed a
#: URL and no credentials -- MLflow's remote-tracking guidance, which the mount already
#: supports by proxying artifacts.
KERNEL_TRACKING_URI_ENV = "NOTEBOOK_MLFLOW_TRACKING_URI"
#: Optional bearer credential for that URL, so gating the mount later needs no change
#: here. Passed on as MLflow's own ``MLFLOW_TRACKING_TOKEN``.
KERNEL_TRACKING_BEARER_ENV = "NOTEBOOK_MLFLOW_TRACKING_TOKEN"

#: Where logged model artifacts live. An object-store URI in production, because a
#: container's filesystem does not outlive the container; the project cache otherwise.
ARTIFACT_ROOT_ENV = "MLFLOW_ARTIFACT_ROOT"

#: Slack over a run's own budget before the worker is treated as wedged. The budget
#: bounds the *search*; registering the model and computing SHAP come after it.
_TRAIN_TIMEOUT_SLACK = 900.0


def _train_out_of_process(
    rows: list[dict[str, Any]], *, budget: float, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Run one AutoML search in a child process; return its report dict.

    An OOM kills the child, so the caller sees a failed job rather than the service
    restarting, and the reply names the cause: "the process running this job restarted"
    points the reader at the app rather than at the run.
    """
    request = json.dumps({"rows": rows, "kwargs": kwargs})
    try:
        done = subprocess.run(
            [sys.executable, "-m", "elbi._train_worker"],
            input=request,
            capture_output=True,
            text=True,
            timeout=budget + _TRAIN_TIMEOUT_SLACK,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelError(
            f"training exceeded its {budget:.0f}s budget by more than "
            f"{_TRAIN_TIMEOUT_SLACK:.0f}s and was stopped"
        ) from exc
    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {done.returncode}"
        if done.returncode in (-9, 137):
            raise ModelError(
                "training ran out of memory and was killed. It runs in its own "
                "process, so the app is unaffected; retry with fewer features, a "
                f"smaller budget, or more memory for the container. ({tail})"
            )
        raise ModelError(f"the training process failed: {tail}")
    try:
        reply = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ModelError("the training process returned no result") from exc
    if "error" in reply:
        raise ModelError(str(reply["error"]))
    return dict(reply["report"])


def kernel_tracking_env() -> dict[str, str]:
    """The MLflow variables a notebook kernel starts with, empty when unconfigured.

    The kernel worker reads these to decide whether registering a model from a cell
    works or refuses with the reason.
    """
    uri = os.environ.get(KERNEL_TRACKING_URI_ENV, "").strip()
    if not uri:
        return {}
    env = {"MLFLOW_TRACKING_URI": uri, "MLFLOW_REGISTRY_URI": uri}
    bearer = os.environ.get(KERNEL_TRACKING_BEARER_ENV, "").strip()
    if bearer:
        env["MLFLOW_TRACKING_TOKEN"] = bearer
    return env


def _store_uri(uri: str | None) -> str | None:
    """A tracking URI SQLAlchemy can open, on the driver MLflow's store requires.

    MLflow compares a version *string* against an INTEGER column -- its client signature
    asks for a string and its REST handlers pass one -- so it needs a driver that leaves
    parameters untyped for Postgres to coerce. psycopg 3 types them, and aliases then
    fail with ``operator does not exist: integer = character varying``; psycopg2 is what
    MLflow documents and tests, so the store is pinned to it here even though the app's
    own tables use psycopg 3. Bare ``postgres://`` is spelled out for SQLAlchemy 2,
    which dropped that alias. An http(s) server and a sqlite file are left alone, and
    so is a URI that already names a driver -- that one is the operator's choice.
    """
    if uri and uri.startswith(("postgres://", "postgresql://")):
        return "postgresql+psycopg2://" + uri.split("://", 1)[1]
    return uri


def make_model_service(
    *,
    store: Store | None,
    cache_dir: Path,
    load_datasets: Callable[[], dict[str, Rows]],
    list_derivations: Callable[[], list[str]] | None = None,
    load_derivation_rows: Callable[[str], list[dict[str, Any]]] | None = None,
    apply_derivation: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
    | None = None,
    list_training_sets: Callable[[], list[str]] | None = None,
    load_training_set_rows: Callable[[str], list[dict[str, Any]]] | None = None,
) -> ModelService | None:
    """A :class:`ModelService` for the project, or ``None`` without the ml extra.

    Missing dependencies are the one expected failure (the extra is optional); the
    app then runs without model surfaces and the chat tools decline with the
    install hint instead.

    The three derivation seams make certified derivations first-class training
    sources: ``list_derivations`` names them, ``load_derivation_rows`` computes
    (or reads the cached) output of one, and ``apply_derivation`` runs one over
    caller-supplied rows (the skew-proof scoring transform). The two training-set
    seams do the same for a feature store's materialized point-in-time joins:
    ``list_training_sets`` names them and ``load_training_set_rows`` reads one. All
    optional; a bare service trains on bound datasets only.
    """
    try:
        from elbi_core.ml import require_ml

        require_ml()
    except ModelError:
        return None

    def resolve_uri() -> str:
        configured = configured_tracking_uri(store) if store is not None else None
        return _store_uri(configured) or f"sqlite:///{cache_dir / 'mlflow.db'}"

    return ModelService(
        resolve_tracking_uri=resolve_uri,
        # An object-store URI (``s3://bucket/mlartifacts``) is the production shape and
        # the only one that survives a restart; a local path is the laptop default.
        artifact_root=os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
        or cache_dir / "mlartifacts",
        load_datasets=load_datasets,
        list_derivations=list_derivations,
        load_derivation_rows=load_derivation_rows,
        apply_derivation=apply_derivation,
        list_training_sets=list_training_sets,
        load_training_set_rows=load_training_set_rows,
    )


def mount_mlflow_ui(app: Any, service: ModelService) -> bool:
    """Mount the real MLflow UI and REST API at ``/mlflow``, on the same store.

    This is MLflow's own server application (the Flask app ``mlflow server``
    runs under gunicorn), not an imitation, so experiments, run charts, model
    pages, and comparisons are all there. It binds the tracking store resolved at
    startup; changing the store setting needs a restart, which the settings page
    already says about the export URI. Returns whether the mount happened (the
    server module needs the full mlflow package).

    It ships unauthenticated and is served that way, like the rest of the app: put
    the deployment behind something that authenticates if it is reachable by anyone
    but you. Artifacts are proxied through the server, so reaching ``/mlflow`` reaches
    everything the tracking store holds.
    """
    import os

    # MLflow's own DNS-rebinding guard only knows localhost hosts, and it is
    # snapshotted when mlflow.server builds its app at import time. Mounted inside
    # this app it never owns a listener, so host validation (and any auth) belongs
    # to the outer server and the inner check must not reject deployed hostnames.
    os.environ.setdefault("MLFLOW_SERVER_ALLOWED_HOSTS", "*")
    try:
        import mlflow.server as mlflow_server
        from mlflow.utils.server_cli_utils import resolve_default_artifact_root
    except ImportError:
        return False
    root = service.artifact_root
    # The server app reads its stores from these env vars (how the mlflow CLI
    # configures the very same app before handing it to gunicorn); unlike the
    # host allowlist they are read per request, so setting them after the import
    # is safe.
    os.environ[mlflow_server.BACKEND_STORE_URI_ENV_VAR] = service.tracking_uri()
    os.environ[mlflow_server.REGISTRY_STORE_URI_ENV_VAR] = service.tracking_uri()
    # The store is where the *server* writes; what clients are told is the proxy scheme.
    # Naming the store as the default artifact root instead hands every client the raw
    # ``s3://`` location, so a notebook kernel uploads straight to the bucket and needs
    # write credentials of its own -- and the ones a kernel is vended are read-only, on
    # purpose. This is the split ``mlflow server --serve-artifacts`` makes, via its own
    # resolver so the two stay in step.
    os.environ[mlflow_server.ARTIFACTS_DESTINATION_ENV_VAR] = _artifact_uri(root)
    os.environ[mlflow_server.SERVE_ARTIFACTS_ENV_VAR] = "true"
    os.environ[mlflow_server.ARTIFACT_ROOT_ENV_VAR] = resolve_default_artifact_root(
        True, "", service.tracking_uri()
    )
    os.environ[mlflow_server.STATIC_PREFIX_ENV_VAR] = "/mlflow"
    _resolve_proxied_artifacts_locally(_artifact_uri(root))
    with warnings.catch_warnings():
        # Starlette's WSGI bridge is deprecated in favor of a2wsgi; it still
        # ships and works, and using it avoids a new dependency. Revisit if a
        # release actually drops it.
        warnings.simplefilter("ignore")
        from starlette.middleware.wsgi import WSGIMiddleware

    def _env_guarded(environ: Any, start_response: Any) -> Any:
        # The server's store getters configure themselves by mutating MLflow's
        # process-global config: set_tracking_uri and set_registry_uri, each of which
        # writes a module global AND an environment variable. In a standalone server
        # (its own process) that is harmless; embedded here it would outlive the request
        # and silently redirect every later MLflow caller in the process (the app,
        # training, scoring) to this server's store. Snapshot the four pieces of global
        # state and restore them when the request returns, isolating the mount as an
        # out-of-process server would be.
        from mlflow.tracking._model_registry import utils as registry_utils
        from mlflow.tracking._tracking_service import utils as tracking_utils

        saved_globals = (tracking_utils._tracking_uri, registry_utils._registry_uri)
        saved_env = {k: os.environ.get(k) for k in _MLFLOW_URI_ENV_VARS}
        try:
            return mlflow_server.app(environ, start_response)
        finally:
            tracking_utils._tracking_uri, registry_utils._registry_uri = saved_globals
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    app.mount("/mlflow", WSGIMiddleware(_env_guarded))
    return True


def _resolve_proxied_artifacts_locally(destination: str) -> None:
    """Read ``mlflow-artifacts:`` URIs from the store this process serves them from.

    That scheme means "ask the tracking server", so MLflow resolves it by calling one
    over HTTP and refuses when the tracking URI is a database. Here the server *is* this
    process: it holds the destination and its credentials, so loading a registered model
    for scoring reads the object store rather than calling back through the app's own
    socket. Kernels are unaffected -- their tracking URI is the mount's URL and they
    keep proxying, which is the whole point of vending them no credentials.
    """
    from mlflow.store.artifact.artifact_repository_registry import (
        _artifact_repository_registry,
        get_artifact_repository,
    )

    base = destination.rstrip("/")

    def _repository(
        artifact_uri: str,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> Any:
        path = artifact_uri.split(":", 1)[1].lstrip("/")
        return get_artifact_repository(f"{base}/{path}" if path else base)

    _artifact_repository_registry.register("mlflow-artifacts", _repository)


def _artifact_uri(root: str | Path) -> str:
    """The artifact root as a URI MLflow accepts, creating it only if it is local.

    An object-store URI is the production shape -- MLflow's documented pattern is a
    database backend with artifacts in object storage -- and it is also what survives a
    restart, since a container's own filesystem does not. A local path stays supported
    for a laptop, and is the only case with a directory to make.
    """
    text = str(root)
    if "://" in text:
        return text
    path = Path(text)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


class ModelService:
    """The app's model lifecycle: train, list, promote, and score.

    The tracking URI is resolved per call (not captured) so a settings change takes
    effect without a restart, the same behavior as the certified-run export. The
    registry client is rebuilt when the URI changes, which also drops its loaded
    model cache (models from the old store must not serve the new one).
    """

    def __init__(
        self,
        *,
        resolve_tracking_uri: Callable[[], str],
        artifact_root: str | Path,
        load_datasets: Callable[[], dict[str, Rows]],
        list_derivations: Callable[[], list[str]] | None = None,
        load_derivation_rows: Callable[[str], list[dict[str, Any]]] | None = None,
        apply_derivation: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
        | None = None,
        list_training_sets: Callable[[], list[str]] | None = None,
        load_training_set_rows: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._resolve_tracking_uri = resolve_tracking_uri
        self._artifact_root = artifact_root
        self._load_datasets = load_datasets
        self._list_derivations = list_derivations
        self._load_derivation_rows = load_derivation_rows
        self._apply_derivation = apply_derivation
        self._list_training_sets = list_training_sets
        self._load_training_set_rows = load_training_set_rows
        self._registry: Any = None
        self._registry_uri: str | None = None

    def tracking_uri(self) -> str:
        """The tracking store currently in effect (env, setting, or local)."""
        return self._resolve_tracking_uri()

    @property
    def artifact_root(self) -> str | Path:
        """Where new experiments' artifacts land for the local store."""
        return self._artifact_root

    def registry(self) -> Any:
        """The registry client for the currently configured tracking store."""
        from elbi_core.ml import ModelRegistry

        uri = self._resolve_tracking_uri()
        if self._registry is None or uri != self._registry_uri:
            self._registry = ModelRegistry(uri)
            self._registry_uri = uri
        return self._registry

    # -- training (shared by the chat tool and the UI's training jobs) ------------

    def feature_sources(self) -> list[dict[str, str]]:
        """Everything a model trains on: datasets, derivations, then training sets."""
        sources = [
            {"name": name, "kind": "dataset"} for name in sorted(self._load_datasets())
        ]
        if self._list_derivations is not None:
            sources += [
                {"name": name, "kind": "derivation"}
                for name in sorted(self._list_derivations())
            ]
        if self._list_training_sets is not None:
            sources += [
                {"name": name, "kind": "training_set"}
                for name in self._list_training_sets()
            ]
        return sources

    def load_source(self, kind: str, name: str) -> list[dict[str, Any]]:
        """The rows of a training source: a bound dataset or a certified derivation.

        A derivation source runs through the project runner (cached and
        content-addressed, so an unchanged pipeline costs a cache read) and must
        produce tabular rows.
        """
        if kind == "dataset":
            rows = self._load_datasets().get(name)
            if not rows:
                available = ", ".join(sorted(self._load_datasets())) or "(none)"
                raise ModelError(f"no dataset named {name!r}. Available: {available}")
            return list(rows)
        if kind == "derivation":
            if self._load_derivation_rows is None:
                raise ModelError("derivation sources are not available here")
            return self._load_derivation_rows(name)
        if kind == "training_set":
            if self._load_training_set_rows is None:
                raise ModelError("training-set sources are not available here")
            rows = self._load_training_set_rows(name)
            if not rows:
                raise ModelError(f"training set {name!r} is empty or does not exist")
            return rows
        raise ModelError(f"unknown source kind {kind!r}")

    def check_train(
        self, name: str, dataset: str, target: str, source_kind: str = "dataset"
    ) -> Rows:
        """Validate a training request cheaply, returning the source's rows.

        Raised errors are the caller-facing kind (a bad name, an unknown source),
        so the UI route can answer 400 before a job is enqueued and the chat tool
        can hand the model the fix.
        """
        if not _NAME_RE.match(name):
            raise ModelError(
                "model names are 1-64 characters: letters, digits, and ._- "
                "(starting alphanumeric)"
            )
        rows = self.load_source(source_kind, dataset)
        if target and target not in rows[0]:
            raise ModelError(f"target column {target!r} is not in {dataset!r}")
        return rows

    def train_report(
        self,
        name: str,
        dataset: str,
        target: str,
        features: Sequence[str],
        task: str,
        time_budget: float,
        metric: str | None,
        max_budget: float = _MAX_JOB_BUDGET,
        ensemble: bool = False,
        time_col: str | None = None,
        horizon: int | None = None,
        groups: str | None = None,
        source_kind: str = "dataset",
        engine: str = "flaml",
    ) -> dict[str, Any]:
        """Run the full train-register loop; raises :class:`ModelError` on failure.

        Returns a dict, not a ``TrainingReport``: the search runs in a child process
        (see :mod:`elbi._train_worker`) and what crosses back is JSON.
        """
        rows = self.check_train(name, dataset, target, source_kind)
        budget = min(max(time_budget, _MIN_BUDGET), max_budget)
        report = _train_out_of_process(
            rows,
            budget=budget,
            kwargs={
                "name": name,
                "target": target,
                "features": list(features),
                "task": task,
                "time_budget": budget,
                "metric": metric,
                "tracking_uri": self._resolve_tracking_uri(),
                "artifact_root": str(self._artifact_root),
                "ensemble": ensemble,
                "dataset": dataset,
                "dataset_kind": source_kind,
                "time_col": time_col,
                "horizon": horizon,
                "groups": groups,
                "engine": engine,
            },
        )
        return {
            **report,
            "source": dataset,
            "source_kind": source_kind,
            "engine": engine,
        }

    def train_result(
        self,
        name: str,
        dataset: str,
        target: str,
        features: Sequence[str],
        task: str,
        time_budget: float,
        metric: str | None,
        ensemble: bool = False,
        time_col: str | None = None,
        horizon: int | None = None,
        groups: str | None = None,
        source_kind: str = "dataset",
        engine: str = "flaml",
    ) -> dict[str, Any]:
        """Train and return the report as a JSON-safe dict (a job's result)."""
        return self.train_report(
            name,
            dataset,
            target,
            features,
            task,
            time_budget,
            metric,
            ensemble=ensemble,
            time_col=time_col,
            horizon=horizon,
            groups=groups,
            source_kind=source_kind,
            engine=engine,
        )

    # -- batch scoring -------------------------------------------------------------

    def batch_score_result(
        self,
        name: str,
        dataset: str,
        ref: str | None = None,
        limit: int | None = None,
        source_kind: str = "dataset",
    ) -> dict[str, Any]:
        """Score a whole source; the full CSV rides on a run, the summary here."""
        from elbi_core.ml import batch_score

        rows = self.load_source(source_kind, dataset)
        report = batch_score(
            rows,
            name=name,
            dataset=dataset,
            tracking_uri=self._resolve_tracking_uri(),
            ref=ref,
            limit=limit,
        )
        return {
            "name": report.name,
            "version": report.version,
            "dataset": report.dataset,
            "n_rows": report.n_rows,
            "run_id": report.run_id,
            "artifact": report.artifact,
            "stats": report.stats,
            "sample": report.sample,
            "rendered": report.render(),
        }

    def render_batch_score(self, ref: str, source: str, kind: str = "dataset") -> str:
        """The chat adapter for batch scoring: rendered summary or an error."""
        name, sep, suffix = ref.partition("@") if "@" in ref else ref.partition("/")
        try:
            result = self.batch_score_result(
                name, source, suffix if sep else None, source_kind=kind
            )
        except ModelError as exc:
            return f"error:\n{exc}"
        rendered: str = result["rendered"]
        return rendered

    def invoke_raw(self, name: str, payload: Any) -> dict[str, Any]:
        """Score RAW rows by first applying the model's feature derivation.

        The skew-proof path: the same certified code that engineered the training
        features engineers the request's features, then the engineered rows score
        exactly as ``/invocations`` would. Only models trained on a derivation
        source have this path; the derivation to apply is read from the version's
        lineage tags, never guessed.
        """
        from elbi_core.ml import parse_invocations, predictions_payload

        if self._apply_derivation is None:
            raise ModelError("feature application is not available here")
        registry = self.registry()
        version = registry.resolve(name)
        versions = {v.version: v for v in registry.versions(name)}
        tags = versions[version].tags if version in versions else {}
        feature_derivation = tags.get("elbi.feature_derivation")
        if not feature_derivation:
            raise ModelError(
                f"model {name!r} v{version} was not trained on a feature "
                "derivation; send engineered features to /invocations instead"
            )
        raw_frame, _ = parse_invocations(payload)
        raw_rows = raw_frame.to_dict(orient="records")
        engineered = self._apply_derivation(feature_derivation, raw_rows)
        if not engineered:
            raise ModelError(
                f"feature derivation {feature_derivation!r} produced no rows "
                "for this input"
            )
        model = registry.load(name, str(version))
        frame, _ = parse_invocations(
            {"dataframe_records": engineered},
            schema=model.metadata.get_input_schema(),
        )
        try:
            predictions = model.predict(frame)
        except Exception as exc:
            raise ModelError(f"scoring failed: {exc}") from exc
        return {
            "features_applied": feature_derivation,
            "n_raw_rows": len(raw_rows),
            "n_scored_rows": len(engineered),
            "engineered": engineered[:_ENGINEERED_SAMPLE],
            **predictions_payload(predictions),
        }

    # -- drift monitoring ----------------------------------------------------------

    def training_script(self, name: str, version: str | None = None) -> str | None:
        """The editable training script logged with a model version, or ``None``.

        Every training run logs ``training_code.py`` (the glass-box script that
        reproduces the winning configuration), so a notebook can open it to re-run or
        adapt the training. ``version`` defaults to the champion (or newest).
        """
        import mlflow

        from elbi_core.ml.training import scoped_tracking

        registry = self.registry()
        resolved = str(version or registry.resolve(name))
        run_id = next(
            (v.run_id for v in registry.versions(name) if str(v.version) == resolved),
            "",
        )
        if not run_id:
            return None
        with scoped_tracking(self._resolve_tracking_uri()):
            try:
                text = mlflow.artifacts.load_text(f"runs:/{run_id}/training_code.py")
            except Exception:
                return None
        return str(text)

    def drift_result(
        self, name: str, current_rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Compare recent traffic against the champion's training reference.

        The reference sample was logged at training time (``reference.csv``); the
        comparison is Evidently's data-drift preset; the outcome is recorded as a
        drift-check run in the model's experiment (summary metrics plus the full
        HTML report as an artifact) so drift history lives with the model.
        """
        import mlflow

        from elbi_core.ml import data_drift
        from elbi_core.ml.training import scoped_tracking

        registry = self.registry()
        version = registry.resolve(name)
        run_id = next(
            (v.run_id for v in registry.versions(name) if v.version == version),
            "",
        )
        if not run_id:
            raise ModelError(f"model {name!r} v{version} has no training run")
        uri = self._resolve_tracking_uri()
        with scoped_tracking(uri):
            try:
                reference_csv = mlflow.artifacts.load_text(
                    f"runs:/{run_id}/reference.csv"
                )
            except Exception as exc:
                raise ModelError(
                    f"model {name!r} v{version} has no training reference sample "
                    "(retrain to enable drift monitoring)"
                ) from exc
        import io

        import pandas as pd

        reference = pd.read_csv(io.StringIO(reference_csv))
        model = registry.load(name, str(version))
        schema = model.metadata.get_input_schema()
        columns = [c["name"] for c in schema.to_dict()] if schema else None
        report = data_drift(reference, list(current_rows), columns=columns)

        with scoped_tracking(uri):
            experiment = mlflow.get_experiment_by_name(name)
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name="drift-check",
                tags={
                    "elbi.drift_check": "true",
                    "elbi.model_version": str(version),
                },
            ) as active:
                mlflow.log_metrics(
                    {
                        "n_drifted": report.n_drifted,
                        "share_drifted": report.share_drifted,
                        "dataset_drift": int(report.dataset_drift),
                        "n_current_rows": len(current_rows),
                    }
                )
                if report.html:
                    mlflow.log_text(report.html, "drift_report.html")
                drift_run = active.info.run_id
        return {
            "name": name,
            "version": version,
            "n_current_rows": len(current_rows),
            "run_id": drift_run,
            **report.summary(),
        }

    # -- chat workspace capabilities (rendered strings the model reads) ----------

    def train(
        self,
        name: str,
        dataset: str,
        target: str,
        features: Sequence[str],
        task: str,
        time_budget: float,
        metric: str | None,
        ensemble: bool = False,
    ) -> str:
        """Train, register, and report; errors come back as text the model can fix."""
        try:
            report = self.train_report(
                name,
                dataset,
                target,
                features,
                task,
                time_budget,
                metric,
                max_budget=INLINE_TRAIN_BUDGET,
                ensemble=ensemble,
            )
        except ModelError as exc:
            return f"error:\n{exc}"
        rendered = str(report["rendered"])
        return rendered + (
            f"\n\nServing: POST /api/serving/{report['name']}/invocations with an "
            'MLflow scoring payload (e.g. {"dataframe_records": [...]}).'
        )

    def render_models(self) -> str:
        """The registry as compact markdown for the chat."""
        try:
            models = self.registry().models()
        except ModelError as exc:
            return f"error:\n{exc}"
        if not models:
            return "No registered models yet. train_model creates one."
        lines = [
            "| model | latest version | champion | description |",
            "| --- | --- | --- | --- |",
        ]
        for m in models:
            champion = f"v{m.champion_version}" if m.champion_version else "(none)"
            lines.append(
                f"| {m.name} | v{m.latest_version} | {champion} "
                f"| {m.description or ''} |"
            )
        return "\n".join(lines)

    def render_predict(
        self, ref: str, rows: Sequence[Mapping[str, Any]], raw: bool = False
    ) -> str:
        """Score ``rows`` with the referenced model and render the predictions.

        ``raw`` routes through the model's recorded feature derivation first (the
        skew-proof path), so the caller sends source-shaped records.
        """
        if not ref.strip():
            return (
                "error:\npredict needs a model reference "
                "(name, name@alias, or name/version)."
            )
        if raw:
            name = ref.partition("@")[0].partition("/")[0].strip()
            try:
                result = self.invoke_raw(name, {"dataframe_records": list(rows)})
            except ModelError as exc:
                return f"error:\n{exc}"
            values = result.get("predictions")
            return (
                f"Applied feature derivation {result['features_applied']!r} to "
                f"{result['n_raw_rows']} raw rows, then scored: {values}"
            )
        try:
            name, version, model = self._resolve_and_load(ref.strip())
            predictions = _predict_frame(model, rows)
        except ModelError as exc:
            return f"error:\n{exc}"
        rendered = ", ".join(str(p) for p in predictions[:50])
        overflow = (
            "" if len(predictions) <= 50 else f" (first 50 of {len(predictions)})"
        )
        return f"Predictions from {name} v{version}{overflow}: [{rendered}]"

    def render_promote(self, name: str, version: int, alias: str) -> str:
        """Move ``alias`` to ``version`` and confirm, or say why it cannot move."""
        try:
            self.registry().promote(name, version, alias)
        except ModelError as exc:
            return f"error:\n{exc}"
        return f"{name} v{version} is now `@{alias}`."

    # -- JSON API views -----------------------------------------------------------

    def models_view(self) -> list[dict[str, Any]]:
        """Every registered model, for ``GET /api/registry/models``."""
        return [
            {
                "name": m.name,
                "description": m.description,
                "latest_version": m.latest_version,
                "champion_version": m.champion_version,
                "created_at_ms": m.created_at_ms,
                "updated_at_ms": m.updated_at_ms,
                "tags": m.tags,
            }
            for m in self.registry().models()
        ]

    def model_view(self, name: str) -> dict[str, Any]:
        """One model with all its versions, for ``GET /api/models/{name}``."""
        versions = self.registry().versions(name)
        champion = next((v.version for v in versions if "champion" in v.aliases), None)
        return {
            "name": name,
            "champion_version": champion,
            "versions": [
                {
                    "version": v.version,
                    "run_id": v.run_id,
                    "experiment_id": v.experiment_id,
                    "created_at_ms": v.created_at_ms,
                    "aliases": list(v.aliases),
                    "description": v.description,
                    "metrics": v.metrics,
                    "params": v.params,
                    "tags": v.tags,
                    # The oracle's judgement of this version, promoted out of the tags
                    # it is recorded under: a reader asking why a version was not made
                    # champion should not have to know where it is kept.
                    "verdict": v.tags.get("elbi.oracle_verdict"),
                    "verdict_detail": v.tags.get("elbi.oracle_detail"),
                }
                for v in versions
            ],
        }

    def columns_view(self, dataset: str) -> list[dict[str, Any]]:
        """A dataset's columns with a numeric flag, for the train dialog.

        A column is numeric when its first non-empty values all parse as numbers;
        the flag only drives UI hints (target suggestions), never training itself,
        so a cheap sample beats scanning every row.
        """
        rows = self._load_datasets().get(dataset)
        if not rows:
            raise ModelError(f"no dataset named {dataset!r}")
        columns = []
        for column in rows[0]:
            values = [r[column] for r in rows[:200] if r.get(column) not in (None, "")]
            numeric = bool(values)
            for value in values:
                try:
                    float(value)
                except (TypeError, ValueError):
                    numeric = False
                    break
            columns.append({"name": column, "numeric": numeric})
        return columns

    def latest_version(self, name: str) -> int:
        """The newest registered version of ``name``, or 0 when unregistered."""
        try:
            return max(int(v.version) for v in self.registry().versions(name))
        except ModelError:
            return 0

    def invoke(self, name: str, payload: Any, ref: str | None) -> dict[str, Any]:
        """Score an MLflow-protocol payload, for ``POST .../invocations``.

        Parsing casts to the model's input schema through MLflow's own scoring
        server code, so JSON integer-vs-double looseness scores exactly as the
        reference server would instead of tripping strict enforcement. Every
        failure downstream of resolution is caller input, raised as
        :class:`ModelError` so the route answers 400, never a 500.
        """
        from elbi_core.ml import parse_invocations, predictions_payload

        registry = self.registry()
        version = registry.resolve(name, ref)
        model = registry.load(name, str(version))
        frame, params = parse_invocations(
            payload, schema=model.metadata.get_input_schema()
        )
        try:
            if params is not None:
                predictions = model.predict(frame, params=params)
            else:
                predictions = model.predict(frame)
        except Exception as exc:  # pyfunc raises its own types for bad input
            raise ModelError(f"scoring failed: {exc}") from exc
        return predictions_payload(predictions)

    def _resolve_and_load(self, ref: str) -> tuple[str, int, Any]:
        """Split a chat model reference into (name, version, loaded model)."""
        name, sep, suffix = ref.partition("@") if "@" in ref else ref.partition("/")
        registry = self.registry()
        version = registry.resolve(name, suffix if sep else None)
        return name, version, registry.load(name, str(version))


def _predict_frame(model: Any, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Predictions for chat-supplied feature records, as a plain list."""
    from elbi_core.ml import parse_invocations

    frame, _ = parse_invocations(
        {"dataframe_records": list(rows)}, schema=model.metadata.get_input_schema()
    )
    try:
        predictions = model.predict(frame)
    except Exception as exc:  # pyfunc schema enforcement raises its own types
        raise ModelError(f"scoring failed: {exc}") from exc
    if hasattr(predictions, "tolist"):
        return list(predictions.tolist())
    return list(predictions)
