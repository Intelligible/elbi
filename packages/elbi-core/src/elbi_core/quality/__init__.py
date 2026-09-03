"""Verifiable data cleaning: declarative data contracts checked deterministically.

A :class:`~elbi.quality.contract.DataContract` declares the quality bar a
table of rows must satisfy (per-field types and constraints, table keys, referential
integrity, and shape). :func:`~elbi.quality.verify.verify_contract` checks it
and returns a three-valued verdict with typed, located violations, the same shape the
verification oracle speaks. The check catalog runs over an engine-agnostic backend, so
one contract verifies ``list[dict]`` rows today and a dataframe/SQL engine later without
changing the contract. :func:`~elbi.quality.suggest.suggest_contract` profiles a
table and proposes a contract, using confidence bounds rather than sample rates so a
proposed tolerance does not overfit the sample it was drawn from.
"""

from .backend import CleaningBackend, RowsBackend, backend_for, register_backend
from .contract import (
    DATA_CONTRACT_SPEC_VERSION,
    Constraints,
    DataContract,
    FieldSpec,
    ForeignKey,
    Reference,
    TableSpec,
    validate_data_contract,
)
from .profile import ColumnProfile, profile_column, profile_columns
from .report import ClauseResult, ContractReport, Violation
from .suggest import suggest_contract
from .verify import verify_contract

__all__ = [
    "DATA_CONTRACT_SPEC_VERSION",
    "ClauseResult",
    "CleaningBackend",
    "ColumnProfile",
    "Constraints",
    "ContractReport",
    "DataContract",
    "FieldSpec",
    "ForeignKey",
    "Reference",
    "RowsBackend",
    "TableSpec",
    "Violation",
    "backend_for",
    "profile_column",
    "profile_columns",
    "register_backend",
    "suggest_contract",
    "validate_data_contract",
    "verify_contract",
]
