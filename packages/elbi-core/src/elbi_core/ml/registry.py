"""A thin, typed view over the MLflow Model Registry.

The registry itself is MLflow's (versions, aliases, run lineage); this wraps the
``MlflowClient`` surface the platform actually uses so callers (the chat tools, the
app's API, the MCP server) share one vocabulary: list models, list a model's
versions with their run metrics, promote a version to ``champion``, and load a
version for scoring. References follow MLflow 3's convention: a bare name means the
champion alias, digits mean a version number, anything else an alias.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from ..errors import ModelError

#: The alias promotion targets. MLflow 3 replaced registry stages with aliases; one
#: mutable name for "the serving version" is the champion/challenger pattern.
CHAMPION = "champion"


@dataclass(frozen=True)
class RegisteredModelInfo:
    """One registered model: its latest version and where the champion points."""

    name: str
    description: str
    latest_version: int
    champion_version: int | None
    created_at_ms: int
    updated_at_ms: int
    tags: dict[str, str]


@dataclass(frozen=True)
class ModelVersionInfo:
    """One version of a registered model, with its training run's record."""

    name: str
    version: int
    run_id: str
    created_at_ms: int
    aliases: tuple[str, ...]
    description: str
    tags: dict[str, str]
    metrics: dict[str, float]
    params: dict[str, str]
    #: The source run's experiment, carried so a UI can deep-link into MLflow's
    #: own pages (its run route needs the experiment id, not just the run id).
    experiment_id: str = ""


class ModelRegistry:
    """Registry operations against one MLflow tracking/registry store.

    Loaded scoring models are cached per resolved version (a ``models:/name/3`` URI),
    so repeated predictions do not re-deserialize the artifact; an alias move is
    picked up because resolution happens on every call, before the cache.
    """

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self._loaded: dict[str, Any] = {}

    def _client(self) -> Any:
        from mlflow import MlflowClient

        return MlflowClient(tracking_uri=self.tracking_uri)

    def models(self) -> list[RegisteredModelInfo]:
        """Every registered model, newest first."""
        infos = []
        for model in self._client().search_registered_models():
            versions = [int(v.version) for v in model.latest_versions] or [0]
            aliases = {alias: int(v) for alias, v in (model.aliases or {}).items()}
            infos.append(
                RegisteredModelInfo(
                    name=model.name,
                    description=model.description or "",
                    latest_version=max(versions),
                    champion_version=aliases.get(CHAMPION),
                    created_at_ms=int(model.creation_timestamp),
                    updated_at_ms=int(model.last_updated_timestamp),
                    tags=dict(model.tags or {}),
                )
            )
        infos.sort(key=lambda m: m.updated_at_ms, reverse=True)
        return infos

    def versions(self, name: str) -> list[ModelVersionInfo]:
        """Every version of ``name``, newest first, with its run's metrics/params."""
        client = self._client()
        versions = client.search_model_versions(f"name = '{_quote(name)}'")
        if not versions:
            raise ModelError(f"no registered model named {name!r}")
        # Aliases live on the registered model; the version search does not carry
        # them, so they are folded in from the model's own alias map.
        model = client.get_registered_model(name)
        aliased: dict[int, list[str]] = {}
        for alias, at_version in (model.aliases or {}).items():
            aliased.setdefault(int(at_version), []).append(alias)
        infos = []
        for version in versions:
            metrics: dict[str, float] = {}
            params: dict[str, str] = {}
            experiment_id = ""
            if version.run_id:
                # A deleted run leaves the version listable, just without its record.
                with suppress(Exception):
                    run = client.get_run(version.run_id)
                    metrics = {k: float(v) for k, v in run.data.metrics.items()}
                    params = dict(run.data.params)
                    experiment_id = str(run.info.experiment_id)
            infos.append(
                ModelVersionInfo(
                    name=name,
                    version=int(version.version),
                    run_id=version.run_id or "",
                    created_at_ms=int(version.creation_timestamp),
                    aliases=tuple(sorted(aliased.get(int(version.version), ()))),
                    description=version.description or "",
                    tags=dict(version.tags or {}),
                    metrics=metrics,
                    params=params,
                    experiment_id=experiment_id,
                )
            )
        infos.sort(key=lambda v: v.version, reverse=True)
        return infos

    def promote(self, name: str, version: int, alias: str = CHAMPION) -> None:
        """Point ``alias`` at ``version`` (the explicit champion/challenger move)."""
        from mlflow.exceptions import MlflowException

        try:
            self._client().set_registered_model_alias(name, alias, str(version))
        except MlflowException as exc:
            raise ModelError(
                f"cannot promote {name!r} version {version}: {exc.message}"
            ) from exc

    def resolve(self, name: str, ref: str | None = None) -> int:
        """The concrete version number ``ref`` names for ``name``.

        ``None`` means the champion, and *only* the champion: registering a version and
        deploying it are two acts, and the alias is the second -- MLflow's own guidance,
        and Unity Catalog's. Resolving to the newest version instead would serve exactly
        the versions that have not earned the alias from the prediction gate.

        Digits are a version number and anything else is an alias; both stay explicit.
        """
        from mlflow.exceptions import MlflowException

        client = self._client()
        if ref is None:
            try:
                return int(client.get_model_version_by_alias(name, CHAMPION).version)
            except MlflowException:
                versions = client.search_model_versions(f"name = '{_quote(name)}'")
                if not versions:
                    raise ModelError(f"no registered model named {name!r}") from None
                latest = max(int(v.version) for v in versions)
                raise ModelError(
                    f"{name!r} has no @{CHAMPION}: {len(versions)} version(s) are "
                    f"registered (newest is {latest}) but none is deployed. Promote "
                    f"one -- a version earns the alias once the prediction gate finds "
                    f"a real signal -- or name a version explicitly."
                ) from None
        try:
            if ref.isdigit():
                return int(client.get_model_version(name, ref).version)
            return int(client.get_model_version_by_alias(name, ref).version)
        except MlflowException as exc:
            raise ModelError(
                f"cannot resolve model {name!r} ({ref}): {exc.message}"
            ) from exc

    def schema(self, name: str, ref: str | None = None) -> dict[str, Any]:
        """The serving contract of a version: its signature and logged input example.

        This is what a scoring UI needs to build a request: the input columns with
        their types, the output shape, and the example rows logged at training time
        (the load-an-example affordance every MLflow-style serving pane offers).
        Fail-soft on the example: a model logged without one still reports its
        signature.
        """
        from .training import require_ml, scoped_tracking

        require_ml()
        import mlflow

        version = self.resolve(name, ref)
        uri = f"models:/{name}/{version}"
        with scoped_tracking(self.tracking_uri):
            info = mlflow.models.get_model_info(uri)
            example: list[dict[str, Any]] | None = None
            with suppress(Exception):
                loaded = mlflow.models.Model.load(uri).load_input_example(uri)
                if hasattr(loaded, "to_dict"):
                    example = loaded.to_dict(orient="records")
        signature = info.signature
        return {
            "version": version,
            "inputs": signature.inputs.to_dict() if signature else [],
            "outputs": signature.outputs.to_dict() if signature else [],
            "input_example": example,
        }

    def delete_version(self, name: str, version: int) -> None:
        """Delete one version of ``name`` (an alias pointing at it goes with it)."""
        from mlflow.exceptions import MlflowException

        try:
            self._client().delete_model_version(name, str(version))
        except MlflowException as exc:
            raise ModelError(
                f"cannot delete {name!r} version {version}: {exc.message}"
            ) from exc
        self._loaded.pop(f"models:/{name}/{version}", None)

    def delete(self, name: str) -> None:
        """Delete the registered model ``name`` and every version of it."""
        from mlflow.exceptions import MlflowException

        try:
            self._client().delete_registered_model(name)
        except MlflowException as exc:
            raise ModelError(f"cannot delete {name!r}: {exc.message}") from exc
        prefix = f"models:/{name}/"
        for uri in [u for u in self._loaded if u.startswith(prefix)]:
            self._loaded.pop(uri, None)

    def load(self, name: str, ref: str | None = None) -> Any:
        """The pyfunc scoring model for ``name`` at ``ref`` (cached per version)."""
        from .training import require_ml, scoped_tracking

        require_ml()
        import mlflow

        version = self.resolve(name, ref)
        uri = f"models:/{name}/{version}"
        if uri not in self._loaded:
            with scoped_tracking(self.tracking_uri):
                self._loaded[uri] = mlflow.pyfunc.load_model(uri)
        return self._loaded[uri]


def _quote(name: str) -> str:
    """Escape a model name for a registry search filter (single-quoted string)."""
    return name.replace("'", "''")
