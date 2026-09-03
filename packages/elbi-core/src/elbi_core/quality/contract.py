"""The DataContract: the declarative quality bar a table of rows must satisfy.

A contract pairs per-field type and value constraints with table-level keys,
referential integrity, and shape. It is the cleaning counterpart to a derivation's
``claim``: a claim asserts an inferential conclusion the oracle checks for soundness;
a contract asserts *data validity* the checker verifies against the rows. On
construction from a manifest the contract is validated against the bundled Data
Contract Spec, so an ill-formed contract fails loudly rather than at check time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

from ..errors import SpecValidationError

#: The Data Contract Spec version this SDK implements (MAJOR.MINOR).
DATA_CONTRACT_SPEC_VERSION = "1.0"

#: The declared field types. Kept to a core set; each is validated by parsing a
#: present value against it without mutation (see :mod:`._stats`).
FIELD_TYPES = ("string", "integer", "number", "boolean", "date", "datetime")


@dataclass(frozen=True)
class Constraints:
    """Value-level constraints on a single field.

    Which constraints are legal depends on the field type (numeric range only for
    numeric types; length and pattern only for strings); the spec schema enforces
    that. All are optional; an unset constraint is not checked.
    """

    required: bool = False
    unique: bool = False
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    enum: tuple[Any, ...] | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to the spec's ``constraints`` object, omitting unset keys."""
        out: dict[str, Any] = {}
        if self.required:
            out["required"] = True
        if self.unique:
            out["unique"] = True
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.min_length is not None:
            out["minLength"] = self.min_length
        if self.max_length is not None:
            out["maxLength"] = self.max_length
        if self.pattern is not None:
            out["pattern"] = self.pattern
        if self.enum is not None:
            out["enum"] = list(self.enum)
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Constraints:
        """Parse a spec ``constraints`` object."""
        enum = data.get("enum")
        return cls(
            required=bool(data.get("required", False)),
            unique=bool(data.get("unique", False)),
            minimum=_opt_float(data.get("minimum")),
            maximum=_opt_float(data.get("maximum")),
            min_length=_opt_int(data.get("minLength")),
            max_length=_opt_int(data.get("maxLength")),
            pattern=data.get("pattern"),
            enum=tuple(enum) if enum is not None else None,
        )


@dataclass(frozen=True)
class FieldSpec:
    """A column's declared type, constraints, and per-field tolerance."""

    name: str
    type: str
    constraints: Constraints = field(default_factory=Constraints)
    #: Fraction of present values that must satisfy the value constraints (missing
    #: values are judged only by ``required``). Default 1.0 (strict).
    mostly: float = 1.0
    description: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``field`` object."""
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.description:
            out["description"] = self.description
        constraints = self.constraints.to_manifest()
        if constraints:
            out["constraints"] = constraints
        if self.mostly != 1.0:
            out["mostly"] = self.mostly
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> FieldSpec:
        """Parse a spec ``field`` object."""
        return cls(
            name=str(data["name"]),
            type=str(data["type"]),
            constraints=Constraints.from_manifest(data.get("constraints", {})),
            mostly=float(data.get("mostly", 1.0)),
            description=data.get("description"),
        )


@dataclass(frozen=True)
class Reference:
    """The target of a foreign key: a resource and the columns keyed into it."""

    resource: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ForeignKey:
    """A referential-integrity constraint: local columns keyed into a resource."""

    fields: tuple[str, ...]
    reference: Reference

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``foreignKey`` object."""
        return {
            "fields": list(self.fields),
            "reference": {
                "resource": self.reference.resource,
                "fields": list(self.reference.fields),
            },
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> ForeignKey:
        """Parse a spec ``foreignKey`` object."""
        ref = data["reference"]
        return cls(
            fields=tuple(str(c) for c in data["fields"]),
            reference=Reference(
                resource=str(ref["resource"]),
                fields=tuple(str(c) for c in ref["fields"]),
            ),
        )


@dataclass(frozen=True)
class TableSpec:
    """Table-level constraints over the whole set of rows."""

    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    row_count_min: int | None = None
    row_count_max: int | None = None
    columns_match: bool = False

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``table`` object, omitting empty parts."""
        out: dict[str, Any] = {}
        if self.primary_key:
            out["primaryKey"] = list(self.primary_key)
        if self.unique_keys:
            out["uniqueKeys"] = [list(key) for key in self.unique_keys]
        if self.foreign_keys:
            out["foreignKeys"] = [fk.to_manifest() for fk in self.foreign_keys]
        row_count = {}
        if self.row_count_min is not None:
            row_count["min"] = self.row_count_min
        if self.row_count_max is not None:
            row_count["max"] = self.row_count_max
        if row_count:
            out["rowCount"] = row_count
        if self.columns_match:
            out["columnsMatch"] = True
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> TableSpec:
        """Parse a spec ``table`` object."""
        row_count = data.get("rowCount", {})
        return cls(
            primary_key=tuple(str(c) for c in data.get("primaryKey", ())),
            unique_keys=tuple(
                tuple(str(c) for c in key) for key in data.get("uniqueKeys", ())
            ),
            foreign_keys=tuple(
                ForeignKey.from_manifest(fk) for fk in data.get("foreignKeys", ())
            ),
            row_count_min=_opt_int(row_count.get("min")),
            row_count_max=_opt_int(row_count.get("max")),
            columns_match=bool(data.get("columnsMatch", False)),
        )


@dataclass(frozen=True)
class DataContract:
    """A declarative quality bar for a table of rows.

    Construct from a validated manifest with :meth:`from_manifest`, or build
    programmatically and call :meth:`validate` (or run through the authoring loop,
    which validates on the way in).
    """

    fields: tuple[FieldSpec, ...]
    table: TableSpec = field(default_factory=TableSpec)
    description: str | None = None

    def field_names(self) -> tuple[str, ...]:
        """The declared column names, in order."""
        return tuple(f.name for f in self.fields)

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a full, spec-conformant DataContract manifest."""
        out: dict[str, Any] = {
            "specVersion": DATA_CONTRACT_SPEC_VERSION,
            "kind": "DataContract",
            "fields": [f.to_manifest() for f in self.fields],
        }
        if self.description:
            out["description"] = self.description
        table = self.table.to_manifest()
        if table:
            out["table"] = table
        return out

    def validate(self) -> None:
        """Validate this contract against the spec.

        Raises:
            SpecValidationError: if the contract is not spec-conformant, including a
                duplicate field name (an invariant the schema cannot express).
        """
        validate_data_contract(self.to_manifest())

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> DataContract:
        """Parse and validate a DataContract manifest.

        Raises:
            SpecValidationError: if the manifest does not conform to the spec.
        """
        validate_data_contract(data)
        return cls(
            fields=tuple(FieldSpec.from_manifest(f) for f in data["fields"]),
            table=TableSpec.from_manifest(data.get("table", {})),
            description=data.get("description"),
        )


@lru_cache(maxsize=1)
def load_data_contract_schema() -> dict[str, Any]:
    """Return the bundled Data Contract JSON Schema as a dict."""
    resource = files("elbi_core") / "spec" / "data_contract.schema.json"
    schema: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return schema


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = load_data_contract_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_data_contract(manifest: dict[str, Any]) -> None:
    """Validate a DataContract manifest against the spec.

    Raises:
        SpecValidationError: if the manifest does not conform. Field-name uniqueness
            is checked here rather than in the schema, which cannot express it.
    """
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(e.path))
    messages = [_format_error(error) for error in errors]
    messages.extend(_duplicate_field_errors(manifest))
    if messages:
        raise SpecValidationError(messages)


def _duplicate_field_errors(manifest: dict[str, Any]) -> list[str]:
    fields = manifest.get("fields")
    if not isinstance(fields, list):
        return []
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in fields:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str):
            continue
        if name in seen:
            duplicates.append(f"fields: duplicate field name {name!r}")
        seen.add(name)
    return duplicates


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)
