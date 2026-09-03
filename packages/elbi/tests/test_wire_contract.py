"""The API's field names and the TypeScript client's field names, checked against each
other.

Both are held to the same convention: the JSON API is camelCase, except on the routes
`casing._STANDARD_ROUTES` exempts. One test walks the live routes and reads their
payloads; the other reads the client's interfaces and the routes each is fetched from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from elbi.casing import _CAMEL, _SNAKE, _STANDARD_ROUTES, OPAQUE_KEYS

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

WEB = Path(__file__).resolve().parents[1] / "web" / "src"

_SALES = "customer_id,amount\nc1,100\nc2,5\nc3,40\nc4,250\n"


def _served(root: Path) -> Any:
    """The app as `elbi serve` assembles it, over a one-source project."""
    from elbi.serve import build

    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "elbi.yaml").write_text(
        "project: contract\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    return build(root, with_mcp=False)


def _exempt(path: str) -> bool:
    return path.startswith(_STANDARD_ROUTES)


def _offending_keys(payload: object, *, at: str = "") -> list[str]:
    """Every snake_case key in ``payload``, with the path that reaches it.

    Values under `OPAQUE_KEYS` are user column names, and are not descended into.
    """
    if isinstance(payload, list):
        return [
            k
            for i, v in enumerate(payload)
            for k in _offending_keys(v, at=f"{at}[{i}]")
        ]
    if not isinstance(payload, dict):
        return []
    bad = []
    for key, value in payload.items():
        if _SNAKE.match(key):
            bad.append(f"{at}.{key}" if at else key)
        if key not in OPAQUE_KEYS:
            bad += _offending_keys(value, at=f"{at}.{key}" if at else key)
    return bad


def _readable_routes(app: Any) -> list[str]:
    """Every GET route that takes no path parameter, so it can be called as-is."""
    return sorted(
        {
            route.path
            for route in app.routes
            if "GET" in getattr(route, "methods", ())
            and route.path.startswith("/api/")
            and "{" not in route.path
        }
    )


def test_every_camelcased_route_answers_in_camelcase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No route outside the exemption list may put a snake_case field on the wire."""
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    app = _served(tmp_path / "proj")
    checked, offences = 0, {}
    with TestClient(app) as http:
        for path in _readable_routes(app):
            if _exempt(path):
                continue
            response = http.get(path)
            if response.status_code != 200:
                continue  # a route needing configuration this app has none of
            if not response.headers.get("content-type", "").startswith(
                "application/json"
            ):
                continue
            checked += 1
            if bad := _offending_keys(response.json()):
                offences[path] = sorted(set(bad))[:8]
    # A floor near the count this walk reaches today.
    assert checked >= 40, f"only {checked} routes answered; the walk found too little"
    assert not offences, "snake_case on the wire:\n" + json.dumps(offences, indent=2)


def _interfaces(source: str) -> dict[str, list[str]]:
    """Field names per exported object type, in declaration order."""
    return {
        name: [f for f, _type in fields] for name, fields in _declared(source).items()
    }


def _declared(source: str) -> dict[str, list[tuple[str, str]]]:
    """Each exported object type's fields, as (name, declared type)."""
    found: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in source.splitlines():
        if match := re.match(r"export (?:interface|type) (\w+)\b", line):
            current = match.group(1)
            found[current] = []
        elif line.startswith("}"):
            current = None
        elif current and (
            field := re.match(r"\s+([A-Za-z][A-Za-z0-9_]*)\??:\s*(.*?)\s*$", line)
        ):
            found[current].append((field.group(1), field.group(2)))
    return found


def _with_nested(
    routes: dict[str, set[str]], declared: dict[str, list[tuple[str, str]]]
) -> dict[str, set[str]]:
    """Extend ``routes`` to the interfaces reached through a fetched one's field types.

    A nested type is never a type argument of its own, so it arrives on the wire only as
    part of its parent and is held to the parent's convention.
    """
    reached = {name: set(paths) for name, paths in routes.items()}
    changed = True
    while changed:
        changed = False
        for name in list(reached):
            for _field, declared_type in declared.get(name, []):
                for ref in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", declared_type):
                    if ref in declared and not reached[name] <= reached.get(ref, set()):
                        reached.setdefault(ref, set()).update(reached[name])
                        changed = True
    return reached


#: A typed fetch helper call: any identifier, a type argument, then its arguments up to
#: the first `/api/...` literal. The path is the first argument for `json`/`post` and
#: the second for `send(method, path)`, so its position is not fixed.
_FETCH = re.compile(r"\b\w+<([^>()]+)>\([^)]*?[`\"'](/api/[^`\"']*)")

CLIENT_MODULES = sorted(
    p.name
    for p in (Path(__file__).resolve().parents[1] / "web" / "src" / "lib").glob("*.ts")
)


#: Helpers that always GET. `getJson(path, fallback)` carries a fallback argument, so
#: trailing arguments are allowed; `json(path, init)` can carry a method and is only a
#: read when it does not.
_ALWAYS_GET = ("getJson", "getRegistryJson")


def _get_routes(source: str) -> dict[str, set[str]]:
    """The routes each type is fetched from by a GET, keyed by type name.

    `send("POST", path)` and `post(path)` reach paths a GET answers differently -- a
    create and a list share `/api/warehouse/sources` -- so only reads count here.
    """
    routes: dict[str, set[str]] = {}
    for match in re.finditer(
        r"\b(\w+)<([^>()]+)>\(\s*[`\"'](/api/[^`\"']*)[`\"']([^;]*?)\)", source
    ):
        helper, name, path, rest = match.groups()
        reads = helper in _ALWAYS_GET or (helper == "json" and "method" not in rest)
        if not reads:
            continue
        routes.setdefault(
            name.replace("[]", "").replace(" | null", "").strip(), set()
        ).add(path)
    return routes


def _fetch_routes(source: str) -> dict[str, set[str]]:
    """The routes each type is fetched from, by its use as a type argument."""
    routes: dict[str, set[str]] = {}
    for match in _FETCH.finditer(source):
        name = match.group(1).replace("[]", "").replace(" | null", "").strip()
        routes.setdefault(name, set()).add(match.group(2))
    return routes


@pytest.mark.parametrize("module", CLIENT_MODULES)
def test_client_interfaces_match_the_conventions_of_their_routes(module: str) -> None:
    """A snake_case field is a bug unless the interface's route is exempt."""
    source = (WEB / "lib" / module).read_text()
    declared = _declared(source)
    routes = _with_nested(_fetch_routes(source), declared)
    offences: dict[str, object] = {}
    for name, fields in _interfaces(source).items():
        fetched = routes.get(name)
        if not fetched or any(_exempt(route) for route in fetched):
            continue
        if snake := [f for f in fields if _SNAKE.match(f)]:
            offences[name] = {"routes": sorted(fetched), "snake_case fields": snake}
    assert not offences, "TypeScript fields that the API camelCases:\n" + json.dumps(
        offences, indent=2
    )


def _wire_keys(payload: object) -> set[str]:
    """The keys of a payload, or of its items when it is a list."""
    if isinstance(payload, list):
        return {k for item in payload for k in _wire_keys(item)}
    return set(payload) if isinstance(payload, dict) else set()


def test_no_client_type_declares_a_field_the_route_never_sends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required field absent from a live payload is a rename or a leftover.

    Optional fields are skipped: absent is what optional means.
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    app = _served(tmp_path / "proj")
    offences: dict[str, object] = {}
    compared = 0
    with TestClient(app) as http:
        for module in CLIENT_MODULES:
            source = (WEB / "lib" / module).read_text()
            declared = _declared(source)
            # Only types fetched directly: a nested one describes an object inside the
            # payload, not the payload, so its fields are not the top-level keys.
            routes = _get_routes(source)
            for name, fields in declared.items():
                required = [f for f, _t in fields if f"{f}?:" not in source]
                for route in routes.get(name, ()):
                    # Query strings are interpolated by the client; the path answers on
                    # its defaults, which is the shape being compared.
                    path = route.split("?")[0]
                    if "{" in path or "$" in path or _exempt(path):
                        continue
                    response = http.get(path)
                    if response.status_code != 200 or not response.headers.get(
                        "content-type", ""
                    ).startswith("application/json"):
                        continue
                    keys = _wire_keys(response.json())
                    if not keys:
                        continue  # an empty collection describes no shape
                    compared += 1
                    if missing := [f for f in required if f not in keys]:
                        offences[f"{module}:{name}"] = {
                            "route": path,
                            "missing": missing,
                        }
    assert compared >= 18, f"only {compared} client types met a non-empty payload"
    assert not offences, "client fields absent from the wire:\n" + json.dumps(
        offences, indent=2
    )


def test_the_client_half_reaches_most_of_the_declared_interfaces() -> None:
    """The per-module checks pass for free on a type they cannot resolve to a route."""
    total = checked = 0
    for module in CLIENT_MODULES:
        declared = _declared((WEB / "lib" / module).read_text())
        if not declared:
            continue
        routes = _with_nested(
            _fetch_routes((WEB / "lib" / module).read_text()), declared
        )
        total += len(declared)
        checked += sum(
            1
            for name in declared
            if name in routes and not any(_exempt(r) for r in routes[name])
        )
    assert checked >= 125, f"only {checked} of {total} client types resolve to a route"


def test_bare_models_spell_their_fields_the_same_either_way() -> None:
    """`Bare` serves exempt routes too, so its fields must be alias-invariant."""
    from pydantic.alias_generators import to_camel

    from elbi import wire

    checked = 0
    for name in dir(wire):
        model = getattr(wire, name)
        if not (isinstance(model, type) and issubclass(model, wire.Bare)):
            continue
        for field in model.model_fields:
            assert to_camel(field) == field, f"{name}.{field} is not alias-invariant"
            checked += 1
    assert checked >= 2


def test_the_contract_would_catch_a_regression() -> None:
    """The checks fail on the shapes they exist to catch."""
    assert _offending_keys({"maxBudget": 1, "max_budget": 1}) == ["max_budget"]
    assert _offending_keys({"a": [{"data_hash": "x"}]}) == ["a[0].data_hash"]
    # A user column inside an opaque key is data, not a field name.
    assert _offending_keys({"rows": [{"total_spend": 1}]}) == []
    assert _interfaces("export interface A {\n  a_b: string\n}\n") == {"A": ["a_b"]}
    assert _CAMEL.match("maxBudget") and not _CAMEL.match("max_budget")
