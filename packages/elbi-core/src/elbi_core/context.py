"""The execution context passed to a compute function.

A :class:`Context` exposes the derivation's resolved inputs and the parameter
values the agent supplied. Dataset inputs are resolved to a
:class:`~elbi.data.Table`; derivation inputs are resolved to the upstream
:class:`~elbi.artifact.Artifact`; semantic-model inputs are resolved to the
:class:`~elbi.semantic_model.SemanticModel` itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import MissingInputError, ParamError


class Context:
    """Resolved inputs and parameters for a single derivation run."""

    def __init__(
        self,
        inputs: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self._inputs = dict(inputs)
        self._params = dict(params or {})

    def input(self, name: str) -> Any:
        """Return a resolved input by name.

        Raises:
            MissingInputError: if ``name`` is not an input of this derivation.
        """
        if name not in self._inputs:
            available = ", ".join(sorted(self._inputs)) or "(none)"
            raise MissingInputError(
                f"no input named {name!r}; available inputs: {available}"
            )
        return self._inputs[name]

    def param(self, name: str) -> Any:
        """Return a supplied (or defaulted) parameter value by name.

        Raises:
            ParamError: if ``name`` is not a declared parameter of this derivation.
        """
        if name not in self._params:
            available = ", ".join(sorted(self._params)) or "(none)"
            raise ParamError(
                f"no parameter named {name!r}; declared params: {available}"
            )
        return self._params[name]

    @property
    def inputs(self) -> Mapping[str, Any]:
        """A read-only view of all resolved inputs."""
        return dict(self._inputs)

    @property
    def params(self) -> Mapping[str, Any]:
        """A read-only view of all parameter values."""
        return dict(self._params)
