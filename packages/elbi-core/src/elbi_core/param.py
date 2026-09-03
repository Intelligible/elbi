"""Typed parameters an agent supplies when running a derivation.

A derivation may declare parameters; when served over MCP they become the tool's
typed input schema, so an agent can query interactively (e.g. filter by zipcode
or a price band) rather than only read fixed views.

Scalars cover most cases. ``object`` carries a record of named values and
``array`` a list; an array's ``items`` gives its element type. Keep these shallow
(one nesting level): agents fill flat arguments more reliably than nested ones.

Build them with the module-level helpers, mirroring :mod:`serve`::

    param.string(description="Zip code to filter to", required=False)
    param.integer(description="Maximum price", required=False, default=None)
    param.object(description="One feature record to score")
    param.array(items="object", description="A batch of feature records")
"""

from __future__ import annotations

import builtins
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: Scalar parameter types and the structured ``object``/``array`` shapes.
ParamType = Literal["string", "integer", "number", "boolean", "object", "array"]
#: The element type of an ``array`` parameter (no deeper than one nesting level).
ItemType = Literal["string", "integer", "number", "boolean", "object"]

#: Maps a scalar parameter type to the Python type used in the served schema.
PYTHON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class Param:
    """A single typed parameter of a derivation.

    Prefer the :func:`string`, :func:`integer`, :func:`number`, :func:`boolean`,
    :func:`object`, and :func:`array` builders over constructing this directly.
    """

    type: ParamType
    description: str | None = None
    required: bool = True
    default: Any = None
    items: ItemType | None = None

    def __post_init__(self) -> None:
        if self.items is not None and self.type != "array":
            raise ValueError("items is only valid for an 'array' parameter")

    def python_type(self) -> builtins.type:
        """The Python type this parameter maps to."""
        return PYTHON_TYPES[self.type]

    def annotation(self) -> Any:
        """The type annotation a served tool exposes, for the MCP input schema.

        A scalar maps to its Python type, ``object`` to ``dict``, and ``array`` to
        a typed ``list`` (e.g. ``list[float]``), so the schema carries the element
        type the agent should supply.
        """
        if self.type == "array":
            if self.items is None:
                return list
            return types.GenericAlias(list, (PYTHON_TYPES[self.items],))
        return PYTHON_TYPES[self.type]

    def coerce(self, value: Any) -> Any:
        """Coerce a supplied value to this parameter's type.

        A scalar is cast to its Python type; an ``object`` must be a mapping; an
        ``array`` must be a non-string sequence whose elements are each coerced to
        ``items`` when declared.
        """
        if self.type == "object":
            return self._coerce_object(value)
        if self.type == "array":
            return self._coerce_array(value)
        return self._coerce_scalar(value)

    def _coerce_object(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"cannot coerce {value!r} to object (expected a mapping)")
        return dict(value)

    def _coerce_array(self, value: Any) -> list[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"cannot coerce {value!r} to array (expected a sequence)")
        if self.items is None:
            return list(value)
        element = Param(self.items)
        return [element.coerce(item) for item in value]

    def _coerce_scalar(self, value: Any) -> Any:
        target = self.python_type()
        # bool is a subclass of int; keep them distinct.
        already_right_type = isinstance(value, target)
        bool_as_int = target is int and isinstance(value, bool)
        if already_right_type and not bool_as_int:
            return value
        if target is bool:
            return _coerce_bool(value)
        try:
            return target(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot coerce {value!r} to {self.type}") from exc

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to the manifest fragment for this parameter."""
        manifest: dict[str, Any] = {"type": self.type}
        if self.items is not None:
            manifest["items"] = self.items
        if self.description is not None:
            manifest["description"] = self.description
        if not self.required:
            manifest["required"] = False
        if self.default is not None:
            manifest["default"] = self.default
        return manifest


def string(
    *, description: str | None = None, required: bool = True, default: str | None = None
) -> Param:
    """A string parameter."""
    return Param("string", description=description, required=required, default=default)


def integer(
    *, description: str | None = None, required: bool = True, default: int | None = None
) -> Param:
    """An integer parameter."""
    return Param("integer", description=description, required=required, default=default)


def number(
    *,
    description: str | None = None,
    required: bool = True,
    default: float | None = None,
) -> Param:
    """A floating-point parameter."""
    return Param("number", description=description, required=required, default=default)


def boolean(
    *,
    description: str | None = None,
    required: bool = True,
    default: bool | None = None,
) -> Param:
    """A boolean parameter."""
    return Param("boolean", description=description, required=required, default=default)


def object(
    *,
    description: str | None = None,
    required: bool = True,
    default: dict[str, Any] | None = None,
) -> Param:
    """An object parameter: a single record of named values (e.g. one feature row)."""
    return Param("object", description=description, required=required, default=default)


def array(
    *,
    items: ItemType | None = None,
    description: str | None = None,
    required: bool = True,
    default: list[Any] | None = None,
) -> Param:
    """An array parameter: a list of ``items`` (e.g. ids, or a batch of records)."""
    return Param(
        "array",
        items=items,
        description=description,
        required=required,
        default=default,
    )


def _coerce_bool(value: Any) -> bool:
    # ``coerce`` returns real bools directly, so this only sees non-bool values.
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"cannot coerce {value!r} to boolean")
