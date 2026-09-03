"""Tests for project config and data bindings."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import ConfigError, DataBindingError
from elbi_core.config import DataBindings, ProjectConfig


def test_project_config_minimal(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\n", encoding="utf-8")
    config = ProjectConfig.load(path)
    assert config.project == "acme"
    assert config.derivations_dir == "derivations"
    assert config.datasets == ()
    assert config.sources == ()
    assert config.ai_context is None
    assert config.search == "hybrid"
    assert config.sandbox == "subprocess"


def test_sources_parse_csv_shorthand_and_generic(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\n"
        "sources:\n"
        "  - name: sales\n"
        "    type: csv\n"
        "    path: ./fixtures/sales.csv\n"
        "  - name: orders\n"
        "    type: postgres\n"
        "    sync: day\n"
        "    config: { table: public.orders }\n",
        encoding="utf-8",
    )
    sales, orders = ProjectConfig.load(path).sources
    # csv shorthand folds `path` into config and names the table after the source
    assert sales.type == "csv" and sales.sync == "manual"
    assert sales.config == {"path": "./fixtures/sales.csv", "table_name": "sales"}
    # generic form keeps its config and cadence
    assert orders.type == "postgres" and orders.sync == "day"
    assert orders.config == {"table": "public.orders"}


def test_sources_reject_bad_name(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\nsources:\n  - name: 'bad name'\n    type: csv\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="source name"):
        ProjectConfig.load(path)


def test_project_config_search_hybrid(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nsearch: hybrid\n", encoding="utf-8")
    assert ProjectConfig.load(path).search == "hybrid"


def test_project_config_bad_search(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nsearch: fuzzy\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'search' must be"):
        ProjectConfig.load(path)


def test_project_config_sandbox_docker(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nsandbox: docker\n", encoding="utf-8")
    assert ProjectConfig.load(path).sandbox == "docker"


def test_project_config_bad_sandbox(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nsandbox: vm\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'sandbox' must be"):
        ProjectConfig.load(path)


def test_project_config_sandbox_image(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\n", encoding="utf-8")
    assert ProjectConfig.load(path).sandbox_image is None
    path.write_text("project: acme\nsandbox_image: elbi-sandbox\n", encoding="utf-8")
    assert ProjectConfig.load(path).sandbox_image == "elbi-sandbox"


def test_project_config_bad_sandbox_image(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nsandbox_image: '  '\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'sandbox_image' must be"):
        ProjectConfig.load(path)


def test_project_config_egress_default_and_modes(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\n", encoding="utf-8")
    assert ProjectConfig.load(path).egress == "full"
    path.write_text("project: acme\negress: none\n", encoding="utf-8")
    assert ProjectConfig.load(path).egress == "none"
    path.write_text(
        "project: acme\negress: [pypi.org, files.pythonhosted.org]\n", encoding="utf-8"
    )
    assert ProjectConfig.load(path).egress == ("pypi.org", "files.pythonhosted.org")


def test_project_config_bad_egress(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\negress: sometimes\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'egress'"):
        ProjectConfig.load(path)
    path.write_text("project: acme\negress: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'egress'"):
        ProjectConfig.load(path)


def test_project_config_ai_context(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\nai_context: |\n  We sell widgets; revenue is in USD.\n",
        encoding="utf-8",
    )
    assert ProjectConfig.load(path).ai_context == "We sell widgets; revenue is in USD."


def test_project_config_bad_ai_context(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nai_context: [not, a, string]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ai_context"):
        ProjectConfig.load(path)


def test_project_config_full(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\n"
        "derivations_dir: src/derivations\n"
        "datasets:\n"
        "  - sales\n"
        "  - name: customers\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(path)
    assert config.derivations_dir == "src/derivations"
    assert config.dataset_names == ("sales", "customers")


def test_project_config_missing_project(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("derivations_dir: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'project'"):
        ProjectConfig.load(path)


def test_project_config_bad_dataset_name(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\ndatasets:\n  - Bad-Name\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must match"):
        ProjectConfig.load(path)


def test_project_config_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        ProjectConfig.load(path)


def test_project_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        ProjectConfig.load(tmp_path / "absent.yaml")


def test_bindings_absent_file_is_empty(tmp_path: Path) -> None:
    bindings = DataBindings.load(tmp_path / "elbi.dev.yaml")
    assert bindings.bindings == {}


def test_bindings_resolve_relative_path(tmp_path: Path) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text("data:\n  sales: ./fixtures/sales.csv\n", encoding="utf-8")
    bindings = DataBindings.load(path)
    resolved = bindings.resolve("sales", tmp_path)
    assert resolved == (tmp_path / "fixtures" / "sales.csv").resolve()


def test_bindings_resolve_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text("data:\n  sales: env(ACME_DSN)\n", encoding="utf-8")
    abs_csv = Path(tmp_path.anchor) / "abs" / "path.csv"  # absolute on any platform
    monkeypatch.setenv("ACME_DSN", str(abs_csv))
    bindings = DataBindings.load(path)
    assert bindings.resolve("sales", tmp_path) == abs_csv


def test_bindings_resolve_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text("data:\n  sales: env(NOPE_DSN)\n", encoding="utf-8")
    monkeypatch.delenv("NOPE_DSN", raising=False)
    bindings = DataBindings.load(path)
    with pytest.raises(DataBindingError, match="not set"):
        bindings.resolve("sales", tmp_path)


def test_bindings_resolve_unbound(tmp_path: Path) -> None:
    bindings = DataBindings(bindings={})
    with pytest.raises(DataBindingError, match="no local binding"):
        bindings.resolve("sales", tmp_path)


def test_bindings_non_string_value(tmp_path: Path) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text("data:\n  sales: 123\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a string"):
        DataBindings.load(path)


def test_project_config_derivations_dir_not_string(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\nderivations_dir: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'derivations_dir'"):
        ProjectConfig.load(path)


def test_dataset_semantic_model_parses(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\n"
        "datasets:\n"
        "  - bare\n"  # a bare name still works, with an empty model
        "  - name: sales\n"
        "    description: One row per order.\n"
        "    columns:\n"
        "      - name: amount\n"
        "        description: Order total.\n"
        "        type: number\n"
        "        unit: USD\n"
        "        role: measure\n"
        "        synonyms: [revenue, total]\n"
        "        sample_values: ['10.0', '20.0']\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(path)
    assert config.dataset_names == ("bare", "sales")
    assert config.spec_for("bare").columns == ()
    sales = config.spec_for("sales")
    assert sales.description == "One row per order."
    amount = sales.columns[0]
    assert amount.name == "amount"
    assert amount.role == "measure"
    assert amount.unit == "USD"
    assert amount.synonyms == ("revenue", "total")
    assert amount.sample_values == ("10.0", "20.0")


def test_dataset_rejects_unknown_role(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\n"
        "datasets:\n"
        "  - name: sales\n"
        "    columns:\n"
        "      - name: amount\n"
        "        role: confounder\n",  # not a column-intrinsic role
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="role 'confounder' must be one of"):
        ProjectConfig.load(path)


def test_dataset_rejects_duplicate_name(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\ndatasets:\n  - sales\n  - sales\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="declared twice"):
        ProjectConfig.load(path)


def test_dataset_rejects_non_string_synonyms(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: acme\n"
        "datasets:\n"
        "  - name: sales\n"
        "    columns:\n"
        "      - name: amount\n"
        "        synonyms: [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="'synonyms' must be a list of strings"):
        ProjectConfig.load(path)


def test_project_config_datasets_not_a_list(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\ndatasets: not-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'datasets' must be a list"):
        ProjectConfig.load(path)


def test_project_config_dataset_entry_invalid(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: acme\ndatasets:\n  - [a, b]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="name or a mapping"):
        ProjectConfig.load(path)


def test_project_config_empty_file_is_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("", encoding="utf-8")
    # An empty file parses to an empty mapping, which then fails on missing project.
    with pytest.raises(ConfigError, match="'project'"):
        ProjectConfig.load(path)


def test_project_config_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: [unbalanced\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        ProjectConfig.load(path)


def test_bindings_data_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text("data: not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'data' must be a mapping"):
        DataBindings.load(path)


def test_bindings_reject_non_string_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text("data:\n  sales: 123\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="string \\(file\\) or a mapping"):
        DataBindings.load(path)


def test_bindings_parse_sql_mapping(tmp_path: Path) -> None:
    path = tmp_path / "elbi.dev.yaml"
    path.write_text(
        "data:\n  orders:\n    connection: env(DB_URL)\n    query: SELECT 1\n",
        encoding="utf-8",
    )
    bindings = DataBindings.load(path)
    assert bindings.bindings["orders"]["connection"] == "env(DB_URL)"


def test_resolve_rejects_sql_binding(tmp_path: Path) -> None:
    bindings = DataBindings(bindings={"orders": {"connection": "sqlite://"}})
    with pytest.raises(DataBindingError, match="is a SQL binding"):
        bindings.resolve("orders", tmp_path)


def test_load_dataset_sql_roundtrip(tmp_path: Path) -> None:
    sa = pytest.importorskip("sqlalchemy")
    db = f"sqlite:///{tmp_path / 'd.db'}"
    engine = sa.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE t (a INTEGER)"))
        conn.execute(sa.text("INSERT INTO t VALUES (1), (2)"))
    engine.dispose()

    bindings = DataBindings(bindings={"t": {"connection": db, "table": "t"}})
    table = bindings.load_dataset("t", tmp_path)
    assert [r["a"] for r in table.rows] == [1, 2]


def test_load_dataset_sql_resolves_env_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sa = pytest.importorskip("sqlalchemy")
    db = f"sqlite:///{tmp_path / 'd.db'}"
    engine = sa.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE t (a INTEGER)"))
        conn.execute(sa.text("INSERT INTO t VALUES (7)"))
    engine.dispose()

    monkeypatch.setenv("MY_DB_URL", db)
    bindings = DataBindings(
        bindings={"t": {"connection": "env(MY_DB_URL)", "query": "SELECT a FROM t"}}
    )
    assert bindings.load_dataset("t", tmp_path).rows == [{"a": 7}]


def test_sql_binding_unknown_key(tmp_path: Path) -> None:
    bindings = DataBindings(bindings={"t": {"connection": "sqlite://", "qry": "x"}})
    with pytest.raises(ConfigError, match="unknown SQL binding key"):
        bindings.load_dataset("t", tmp_path)


def test_sql_binding_missing_connection(tmp_path: Path) -> None:
    bindings = DataBindings(bindings={"t": {"query": "SELECT 1"}})
    with pytest.raises(ConfigError, match="needs a 'connection'"):
        bindings.load_dataset("t", tmp_path)


def test_sql_binding_bad_max_rows(tmp_path: Path) -> None:
    bindings = DataBindings(
        bindings={"t": {"connection": "sqlite://", "table": "t", "max_rows": 0}}
    )
    with pytest.raises(ConfigError, match="'max_rows' must be a positive int"):
        bindings.load_dataset("t", tmp_path)


def test_sql_binding_bad_timeout(tmp_path: Path) -> None:
    bindings = DataBindings(
        bindings={"t": {"connection": "sqlite://", "table": "t", "timeout": "soon"}}
    )
    with pytest.raises(ConfigError, match="'timeout' must be a positive int"):
        bindings.load_dataset("t", tmp_path)


def test_sql_binding_bad_query_type(tmp_path: Path) -> None:
    bindings = DataBindings(bindings={"t": {"connection": "sqlite://", "query": 5}})
    with pytest.raises(ConfigError, match="'query' must be a string"):
        bindings.load_dataset("t", tmp_path)


def test_sql_binding_bad_table_type(tmp_path: Path) -> None:
    bindings = DataBindings(bindings={"t": {"connection": "sqlite://", "table": 5}})
    with pytest.raises(ConfigError, match="'table' must be a string"):
        bindings.load_dataset("t", tmp_path)


def test_version_file_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "d.csv").write_text("a\n1\n", encoding="utf-8")
    bindings = DataBindings(bindings={"d": "d.csv"})
    v1, preloaded = bindings.version("d", tmp_path)
    assert preloaded is None  # file versioning doesn't load
    (tmp_path / "d.csv").write_text("a\n2\n", encoding="utf-8")
    v2, _ = bindings.version("d", tmp_path)
    assert v1 != v2


def test_version_sql_without_freshness_loads(tmp_path: Path) -> None:
    sa = pytest.importorskip("sqlalchemy")
    db = f"sqlite:///{tmp_path / 'd.db'}"
    engine = sa.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE t (a INTEGER)"))
        conn.execute(sa.text("INSERT INTO t VALUES (1), (2)"))
    engine.dispose()
    bindings = DataBindings(bindings={"t": {"connection": db, "table": "t"}})
    version, preloaded = bindings.version("t", tmp_path)
    assert version
    assert preloaded is not None  # had to load to fingerprint
    assert [r["a"] for r in preloaded.rows] == [1, 2]


def test_version_sql_with_freshness_is_cheap(tmp_path: Path) -> None:
    sa = pytest.importorskip("sqlalchemy")
    db = f"sqlite:///{tmp_path / 'd.db'}"
    engine = sa.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE t (a INTEGER)"))
        conn.execute(sa.text("INSERT INTO t VALUES (1)"))
    engine.dispose()
    bindings = DataBindings(
        bindings={
            "t": {
                "connection": db,
                "table": "t",
                "freshness": "SELECT count(*) AS c FROM t",
            }
        }
    )
    version, preloaded = bindings.version("t", tmp_path)
    assert version
    assert preloaded is None  # freshness probe avoids loading the full table


def test_sql_binding_bad_freshness_type(tmp_path: Path) -> None:
    bindings = DataBindings(
        bindings={"t": {"connection": "sqlite://", "table": "t", "freshness": 5}}
    )
    with pytest.raises(ConfigError, match="'freshness' must be a string"):
        bindings.version("t", tmp_path)


def test_compute_profiles_are_validated_at_load(tmp_path: Path) -> None:
    """A profile menu parses, and a bad one fails when the project loads.

    These become container and pod arguments, so an unvalidated value would be an
    argument-injection vector; and a typo that only surfaced when someone opened a
    notebook would be a poor way to learn about it.
    """
    good = tmp_path / "good.yaml"
    good.write_text(
        "project: t\nsandbox: docker\ncompute_profiles:\n"
        "  - name: small\n    cpu: '2'\n    memory: 4Gi\n"
        "  - name: large\n    cpu: '8'\n    memory: 32Gi\n"
        "default_compute_profile: small\n",
        encoding="utf-8",
    )
    config = ProjectConfig.load(good)
    assert [p["name"] for p in config.compute_profiles] == ["small", "large"]
    assert config.default_compute_profile == "small"

    # Unset is empty, so the deployment's own menu stands.
    plain = tmp_path / "plain.yaml"
    plain.write_text("project: t\n", encoding="utf-8")
    assert ProjectConfig.load(plain).compute_profiles == ()

    for body, expected in [
        ("compute_profiles:\n  - name: small\n    memory: lots\n", "memory quantity"),
        ("compute_profiles:\n  - name: small\n    cpu: 2; rm -rf /\n", "CPU quantity"),
        ("compute_profiles:\n  - name: Small\n", "lowercase alphanumeric"),
        ("compute_profiles:\n  - name: small\n    gpus: 1\n", "unknown profile field"),
        ("compute_profiles: 8g\n", "must be a list"),
    ]:
        bad = tmp_path / "bad.yaml"
        bad.write_text(f"project: t\n{body}", encoding="utf-8")
        with pytest.raises(ConfigError, match=expected):
            ProjectConfig.load(bad)


# --- the policy block (row predicates and column masks) ---


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "elbi.yaml"
    path.write_text(f"project: acme\n{body}", encoding="utf-8")
    return path
