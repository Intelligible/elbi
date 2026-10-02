"""Source configuration types: the schema the connection wizard renders from.

A source declares its identity (name, category, icon) and a list of `fields` describing
the connection form, so the frontend builds the setup UI generically and adding a
connector is config, not new UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# The category rail in the source catalog.
CATEGORIES = [
    "Databases",
    "File storage",
    "Payments & billing",
    "CRM",
    "Marketing & email",
    "Advertising",
    "Analytics",
    "E-commerce",
    "Engineering & monitoring",
    "Productivity",
]

FieldType = Literal["text", "password", "number", "switch", "select", "textarea"]


@dataclass
class SourceField:
    """One input in a source's connection form (a `SourceFieldConfig`)."""

    name: str
    label: str
    type: FieldType = "text"
    required: bool = True
    placeholder: str = ""
    default: Any = None
    # For ``select`` fields: the choices ``[{value,label}]``.
    options: list[dict[str, str]] = field(default_factory=list)
    caption: str = ""
    # A heading this field sits under in the form. Fields sharing one are rendered as a
    # group; an empty section means the field belongs to the connector's main body. Used
    # to keep an optional block -- the SSH tunnel -- from reading as seven more things
    # that have to be filled in.
    section: str = ""
    # The field this one is conditional on. It is shown only when that field holds
    # ``depends_value``, or any truthy value when ``depends_value`` is empty -- and only
    # when that field is itself visible, so a chain of conditions collapses together.
    # An optional block stays out of the way until it is switched on, which is how every
    # comparable product presents one.
    depends_on: str = ""
    depends_value: str = ""
    # When true, the form pairs this text field with a file-upload control that stores
    # the file in the warehouse and fills the field with its path (used by the CSV /
    # Parquet source, so a local file can be uploaded instead of typed as a path/URL).
    upload: bool = False
    # Marks a credential whose ``type`` is not ``password``: a PEM key or a JSON key
    # file is a multi-line document, so it renders as a textarea, but it is every bit
    # as much a secret. A connector declares secrecy here rather than leaving it
    # inferred from the control, so masking on edit and the refusal to store a
    # credential in plaintext both cover it.
    secret: bool = False

    @property
    def is_secret(self) -> bool:
        """Whether this field holds a credential, whatever control it renders as."""
        return self.secret or self.type == "password"


@dataclass
class SourceConfig:
    """A connector's catalog entry + connection-form schema."""

    name: str  # stable source type id, e.g. "postgres", "stripe", "csv"
    label: str
    category: str
    fields: list[SourceField] = field(default_factory=list)
    caption: str = ""
    icon: str = ""  # emoji or icon key; the UI maps it to a glyph
    release_status: Literal["alpha", "beta", "ga"] = "ga"
    docs_url: str = ""
    # A catalogued-but-not-yet-implemented source: shown in the wizard with a "Coming
    # soon" tag and a notify-me action, and not connectable.
    coming_soon: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON form served to the wizard endpoint.

        ``secret`` goes out resolved, so the form reads one flag instead of re-deriving
        secretness from the control type and drifting from the server.
        """
        data = asdict(self)
        for serialized, source_field in zip(data["fields"], self.fields, strict=True):
            serialized["secret"] = source_field.is_secret
        return data


@dataclass
class SourceSchema:
    """A syncable table/endpoint a source exposes (chosen in the wizard)."""

    name: str
    # Columns usable as an incremental cursor (empty ⇒ full-refresh only).
    incremental_fields: list[str] = field(default_factory=list)
    # Whether this table is selected to sync by default.
    default_selected: bool = True
