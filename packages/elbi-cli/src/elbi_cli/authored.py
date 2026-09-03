"""Durable storage for agent-authored derivations.

Accepted agent-authored derivations are persisted as a sidecar under
``.elbi/authored/``, never the project's ``derivations/`` source tree, so
they survive a restart without being written into trusted, importable source.
Each record holds the generated source plus the shape needed to rebuild it; on
load it is reconstructed with ``agent`` origin, so it still runs only under the
sandbox executor and is never imported into the host process. Promoting one into
committed source is a separate, human-driven step.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from elbi_core import Dataset, Derivation, Serve, certify, propose
from elbi_core import serve as serve_builders
from elbi_core.errors import ConfigError
from elbi_core.registry import Registry
from elbi_core.versioning import source_version

#: Directory (under the project's ``.elbi/``) holding authored records.
AUTHORED_DIRNAME = "authored"

_SERVE_BUILDERS: dict[str, Callable[..., Serve]] = {
    "table": serve_builders.table,
    "markdown": serve_builders.markdown,
    "json": serve_builders.json,
    "text": serve_builders.text,
}


class AuthoredStore:
    """Persist and reload agent-authored derivations as a sidecar."""

    def __init__(self, root: Path) -> None:
        self._root = root  # the .elbi/authored directory

    def save(
        self,
        derivation: Derivation,
        attestation: dict[str, Any] | None = None,
        certificate: dict[str, Any] | None = None,
    ) -> None:
        """Persist an agent-authored derivation's source, status, and serve shape.

        ``attestation`` is the verification record for an effect derivation certified
        through the gate; it is stored so the served result still carries its proof
        after a restart. ``certificate`` is that record's signed, exportable envelope,
        stored so the same bytes are re-served after a restart with no re-signing.
        """
        if not derivation.is_agent_authored or derivation.source is None:
            raise ConfigError("only agent-authored derivations carry storable source")
        record: dict[str, Any] = {
            "name": derivation.name,
            "status": derivation.status,
            "source": derivation.source,
            "source_hash": source_version(derivation.source),
            "inputs": {
                key: dataset.name
                for key, dataset in derivation.dataset_inputs().items()
            },
            "serve": _serve_record(derivation.serve),
        }
        if derivation.deps:
            # Persist the sandbox dependencies so a numpy/sklearn derivation still
            # provisions them when reloaded and re-served after a restart.
            record["deps"] = list(derivation.deps)
        if attestation is not None:
            record["attestation"] = attestation
        if certificate is not None:
            record["certificate"] = certificate
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(derivation.name).write_text(
            yaml.safe_dump(record, sort_keys=True), encoding="utf-8"
        )

    def remove(self, name: str) -> None:
        """Delete a stored derivation, live or trashed, if present."""
        self._path(name).unlink(missing_ok=True)
        self._trash_path(name).unlink(missing_ok=True)

    def trash(self, name: str) -> bool:
        """Move a stored derivation's sidecar aside so no loader resurrects it.

        Every loader -- this process's :meth:`load_into`, and the standalone MCP
        server's, which never has a database to check -- globs ``*.yaml`` directly
        under this directory. Renaming into a ``.trash/`` subdirectory is what
        makes a trashed derivation invisible to all of them with no change to any
        loader at all. Returns whether a sidecar was there to move.
        """
        path = self._path(name)
        if not path.exists():
            return False
        self._trash_path(name).parent.mkdir(parents=True, exist_ok=True)
        path.rename(self._trash_path(name))
        return True

    def restore(self, name: str) -> bool:
        """Move a trashed derivation's sidecar back where loaders will find it.

        Returns whether a trashed sidecar was there to move.
        """
        trashed = self._trash_path(name)
        if not trashed.exists():
            return False
        trashed.rename(self._path(name))
        return True

    def read(self, name: str) -> dict[str, Any] | None:
        """Return the stored record for ``name``, or None if absent."""
        path = self._path(name)
        if not path.exists():
            return None
        record: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return record

    def load_one(self, name: str, registry: Registry) -> bool:
        """Reconstruct one stored derivation into ``registry``; whether it loaded.

        The loop body of :meth:`load_into`, factored out so a restore can bring
        back exactly the one derivation that was just untrashed, rather than
        re-walking every sidecar on disk.
        """
        record = self.read(name)
        if record is None or name in registry:
            return False
        inputs = {
            key: Dataset(ref) for key, ref in (record.get("inputs") or {}).items()
        }
        derivation = propose(
            name,
            record["source"],
            serve=_serve_from_record(record.get("serve")),
            inputs=inputs,
            deps=record.get("deps"),
            registry=registry,
        )
        if record.get("status") == "certified":
            certify(derivation, registry=registry)
        return True

    def load_into(self, registry: Registry) -> tuple[str, ...]:
        """Reconstruct stored derivations into ``registry``; return their names.

        Each is rebuilt as a proposed agent derivation (so it routes to the
        sandbox) and certified when it was stored certified. Its source is never
        imported into this process. A name already present (a file-authored
        derivation) takes precedence and is left untouched. A trashed sidecar
        lives under ``.trash/`` and this glob does not descend into it, so a
        trashed derivation never re-registers on a restart with no extra check
        needed here.
        """
        if not self._root.exists():
            return ()
        loaded: list[str] = []
        for path in sorted(self._root.glob("*.yaml")):
            if self.load_one(path.stem, registry):
                loaded.append(path.stem)
        return tuple(loaded)

    def _path(self, name: str) -> Path:
        return self._root / f"{name}.yaml"

    def _trash_path(self, name: str) -> Path:
        return self._root / ".trash" / f"{name}.yaml"


def render_module(record: dict[str, Any], *, promoted_on: str) -> str:
    """Render a stored record as a complete, human-editable derivation module.

    The output is normal source for ``derivations/``: imports, a ``@derivation``
    decorator rebuilt from the record, and the agent's function body. It carries a
    provenance header (deliberately not a ``DO NOT EDIT`` marker, since a promoted
    derivation is now human-owned) and omits ``origin="agent"``, so once discovered
    it is a trusted, human-authored derivation that runs in-process.
    """
    inputs = record.get("inputs") or {}
    serve_record = record.get("serve")
    imports = ["derivation"]
    args: list[str] = []
    if inputs:
        imports.append("Dataset")
        items = ", ".join(f'"{k}": Dataset("{v}")' for k, v in sorted(inputs.items()))
        args.append(f"inputs={{{items}}}")
    if serve_record:
        imports.append("serve")
        title = serve_record.get("title")
        title_arg = f'title="{title}"' if title else ""
        args.append(f"serve=serve.{serve_record['format']}({title_arg})")

    import_line = "from elbi_core import " + ", ".join(sorted(imports))
    if args:
        decorator = "@derivation(\n" + "".join(f"    {arg},\n" for arg in args) + ")"
    else:
        decorator = "@derivation"
    header = (
        "# Origin: AI-generated by the elbi authoring agent; reviewed and\n"
        f"# promoted by a human on {promoted_on}. Now human-maintained, edit freely.\n"
        "# Not auto-regenerated.\n"
    )
    return (
        f"{header}\n"
        "from __future__ import annotations\n\n"
        f"{import_line}\n\n\n"
        f"{decorator}\n"
        f"{record['source']}\n"
    )


def _serve_record(serve: Serve | None) -> dict[str, Any] | None:
    if serve is None:
        return None
    return {"format": serve.format, "title": serve.title}


def _serve_from_record(record: dict[str, Any] | None) -> Serve | None:
    if not record:
        return None
    return _SERVE_BUILDERS[record["format"]](title=record.get("title"))
