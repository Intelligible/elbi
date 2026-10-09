"""Exception hierarchy for the elbi SDK.

All errors raised by the framework derive from :class:`ElbiError`, so
callers can catch the whole family with a single ``except``.
"""

from __future__ import annotations


class ElbiError(Exception):
    """Base class for every error raised by the elbi SDK."""


class SpecValidationError(ElbiError):
    """A manifest failed validation against the Open Derivation Spec."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        joined = "\n  - ".join(messages)
        super().__init__(f"manifest failed spec validation:\n  - {joined}")


class DerivationError(ElbiError):
    """A problem with a derivation definition or its execution."""


class DuplicateDerivationError(DerivationError):
    """Two derivations were registered under the same name."""


class UnknownDerivationError(DerivationError):
    """A derivation was referenced by a name that is not registered."""


class CycleError(DerivationError):
    """The derivation dependency graph contains a cycle."""


class MissingInputError(DerivationError):
    """A required input could not be resolved."""


class ParamError(DerivationError):
    """A supplied parameter is unknown, missing, or the wrong type."""


class AuthorizationError(ElbiError):
    """A caller was not authorized to run or read a derivation."""


class DataBindingError(ElbiError):
    """A dataset could not be bound to a concrete data source."""


class ConfigError(ElbiError):
    """A project or data-bindings configuration file is invalid."""


class CacheError(ElbiError):
    """A cached result could not be stored or loaded."""


class ModelError(ElbiError):
    """A predictive model could not be trained, resolved, or scored."""


class MetricError(ElbiError):
    """A metric could not be resolved (unknown metric, column, or bad query)."""


class CertificateError(ElbiError):
    """A verification certificate is malformed, or its signature does not verify."""


class ComponentError(ElbiError):
    """A components artifact item failed validation against ORC's component schema."""
