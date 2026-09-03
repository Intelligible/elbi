"""Tests for the @derivation decorator and the Derivation model."""

from __future__ import annotations

import pytest

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    SemanticModel,
    SpecValidationError,
    derivation,
    serve,
)
from elbi_core.metrics.osi import OSI_VERSION


def test_decorator_with_args_registers(registry: Registry) -> None:
    @derivation(
        inputs={"sales": Dataset("sales")},
        serve=serve.table(title="Churn"),
        registry=registry,
    )
    def churn_risk(ctx: Context) -> Artifact:
        """Churn risk scores."""
        return Artifact.table([])

    assert churn_risk.name == "churn_risk"
    assert churn_risk.description == "Churn risk scores."
    assert "churn_risk" in registry


def test_explicit_name_overrides_function_name(registry: Registry) -> None:
    @derivation(name="custom", serve=serve.text(), registry=registry)
    def fn(ctx: Context) -> str:
        return "x"

    assert fn.name == "custom"


def test_serve_optional_makes_internal_derivation(registry: Registry) -> None:
    @derivation(registry=registry)
    def fn(ctx: Context) -> str:
        return "x"

    assert fn.serve is None
    assert fn.is_served is False
    # An internal derivation has no serve fragment in its manifest.
    assert "serve" not in fn.to_manifest()


def test_depends_on_inferred_from_derivation_inputs(registry: Registry) -> None:
    @derivation(serve=serve.text(), registry=registry)
    def upstream(ctx: Context) -> str:
        return "u"

    @derivation(
        inputs={"u": upstream},
        depends_on=["upstream"],
        serve=serve.text(),
        registry=registry,
    )
    def downstream(ctx: Context) -> str:
        return "d"

    # Inferred + explicit, deduplicated.
    assert downstream.depends_on == ("upstream",)


def test_manifest_round_trips_through_spec(registry: Registry) -> None:
    @derivation(
        inputs={"sales": Dataset("sales")},
        serve=serve.markdown(title="M"),
        registry=registry,
    )
    def report(ctx: Context) -> str:
        return "body"

    manifest = report.to_manifest()
    assert manifest["kind"] == "Derivation"
    assert manifest["inputs"]["sales"] == {"kind": "dataset", "ref": "sales"}
    assert manifest["serve"] == {"format": "markdown", "title": "M"}
    assert "description" not in manifest  # no docstring -> omitted


def test_invalid_name_fails_at_definition(registry: Registry) -> None:
    with pytest.raises(SpecValidationError):

        @derivation(name="Bad Name", serve=serve.text(), registry=registry)
        def fn(ctx: Context) -> str:
            return "x"


def test_dataset_and_derivation_input_partition(registry: Registry) -> None:
    @derivation(serve=serve.text(), registry=registry)
    def up(ctx: Context) -> str:
        return "u"

    @derivation(
        inputs={"sales": Dataset("sales"), "up": up},
        serve=serve.text(),
        registry=registry,
    )
    def down(ctx: Context) -> str:
        return "d"

    assert set(down.dataset_inputs()) == {"sales"}
    assert set(down.derivation_inputs()) == {"up"}


def _model() -> SemanticModel:
    return SemanticModel.from_osi(
        {
            "version": OSI_VERSION,
            "semantic_model": [
                {
                    "name": "sales_semantics",
                    "datasets": [{"name": "sales", "source": "sales", "fields": []}],
                    "metrics": [
                        {
                            "name": "total_amount",
                            "expression": {
                                "dialects": [
                                    {
                                        "dialect": "ANSI_SQL",
                                        "expression": "SUM(amount)",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    )


def test_semantic_model_input_partitions_apart(registry: Registry) -> None:
    @derivation(serve=serve.text(), registry=registry)
    def up_model(ctx: Context) -> str:
        return "u"

    @derivation(
        inputs={"sales": Dataset("sales"), "up": up_model, "model": _model()},
        serve=serve.text(),
        registry=registry,
    )
    def gated(ctx: Context) -> str:
        return "g"

    assert set(gated.dataset_inputs()) == {"sales"}
    assert set(gated.derivation_inputs()) == {"up"}
    assert set(gated.semantic_model_inputs()) == {"model"}


def test_semantic_model_input_serializes_with_its_own_kind(registry: Registry) -> None:
    @derivation(
        inputs={"sales": Dataset("sales"), "model": _model()},
        serve=serve.text(),
        registry=registry,
    )
    def gated_manifest(ctx: Context) -> str:
        return "g"

    manifest = gated_manifest.to_manifest()
    assert manifest["inputs"] == {
        "sales": {"kind": "dataset", "ref": "sales"},
        "model": {"kind": "semantic_model", "ref": "sales_semantics"},
    }


def test_semantic_model_input_infers_no_dependency(registry: Registry) -> None:
    # A semantic model is static content, so it is not an ordering edge.
    @derivation(
        inputs={"model": _model()},
        serve=serve.text(),
        registry=registry,
    )
    def gated_no_deps(ctx: Context) -> str:
        return "g"

    assert gated_no_deps.depends_on == ()
    assert "dependsOn" not in gated_no_deps.to_manifest()


def test_bare_decorator_form() -> None:
    # @derivation used without calling registers an internal derivation into the
    # default registry; use a unique name to avoid cross-test collisions.
    @derivation
    def bare_internal_node(ctx: Context) -> str:
        return "x"

    assert bare_internal_node.name == "bare_internal_node"
    assert bare_internal_node.is_served is False


def test_defaults_are_human_certified(registry: Registry) -> None:
    @derivation(name="d", serve=serve.text(), registry=registry)
    def d(ctx: Context) -> str:
        return "x"

    assert d.origin == "human"
    assert d.status == "certified"
    assert d.is_certified and not d.is_agent_authored
    # Default provenance is omitted from the manifest (1.0-compatible).
    manifest = d.to_manifest()
    assert "origin" not in manifest and "status" not in manifest


def test_provenance_serialized_when_non_default(registry: Registry) -> None:
    @derivation(
        name="proposed_one",
        serve=serve.text(),
        origin="agent",
        status="proposed",
        registry=registry,
    )
    def proposed_one(ctx: Context) -> str:
        return "x"

    manifest = proposed_one.to_manifest()
    assert manifest["origin"] == "agent"
    assert manifest["status"] == "proposed"
    assert proposed_one.is_agent_authored and not proposed_one.is_certified
