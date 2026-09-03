"""Content fingerprints: the data versions the cache is keyed on.

A derivation's data version is a hash of its code, its parameters, and the
versions of its inputs. When none of those change, the cached result is reused. A
derivation input contributes its output version, enabling early cutoff: a parent
that recomputes to the same output does not invalidate its children.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import inspect
import json
import sys
import textwrap
import types
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def hash_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, hex-encoded."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 of a string."""
    return hash_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    """Canonical JSON text for a value: sorted keys, no whitespace, ASCII escapes.

    The one byte-stable serialization shared by fingerprinting and signing: sorting
    keys and fixing separators makes the output independent of mapping order and
    formatting, so equal content always produces identical bytes. Falls back to
    ``str`` for values JSON cannot encode (e.g. datetimes).
    """
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def hash_json(value: Any) -> str:
    """Stable SHA-256 of a JSON-serializable value (sorted keys).

    Falls back to ``str`` for values JSON cannot encode (e.g. datetimes); this is
    only used for fingerprinting, never for round-tripping.
    """
    return hash_text(canonical_json(value))


def hash_file(path: Path) -> str:
    """SHA-256 of a file's contents, read in chunks (Python 3.10 compatible)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def code_version(fn: Callable[..., Any]) -> str:
    """Return a version string for a function's logic and its dependencies.

    Hashes the function's normalized AST (insensitive to comments and formatting) plus
    the Python minor version, then folds in everything its result can depend on: the
    user code it calls, a function reached by bare name, through a closure, or through a
    module or class attribute (``mod.func``, ``Cls.method``), and the methods of a
    user-defined class it references, recursed cycle-safe; the non-callable module
    globals it reads (a threshold, a lookup table), by value; and, for a third-party
    callable, the version of the distribution that provides it, so a dependency upgrade
    busts the cache.

    The residual is genuinely dynamic dispatch: a call whose target is chosen at run
    time and named nowhere in the source: ``getattr`` with a computed name, a method on
    a value whose type is not referenced statically (only bound at call time), or
    ``eval``/``exec``. Pin an explicit ``code_version`` on the cache policy when such a
    path carries behavior the fingerprint must track. Hashes are comparable only within
    one interpreter minor version and one set of installed dependency versions.
    """
    return _code_version(fn, set())


def _code_version(fn: Callable[..., Any], seen: set[int]) -> str:
    code = getattr(fn, "__code__", None)
    if not isinstance(code, types.CodeType):
        return _opaque(fn)
    if id(code) in seen:  # cycle / already visited
        return ""
    seen.add(id(code))

    py = f"py{sys.version_info.major}.{sys.version_info.minor}"
    global_ns, closure_ns = _referenced_namespaces(fn)
    # Fold module-global *values* (constants, lookup tables) but not closure nonlocals:
    # a nonlocal is definition-site state the compute may mutate as it runs, which would
    # make the version shift mid-run and defeat its own caching.
    parts = [_own_fingerprint(fn), py, *_global_value_parts(global_ns)]
    for callee in _referenced_callables(fn, {**global_ns, **closure_ns}):
        if isinstance(callee, types.CodeType):
            parts.append(repr((callee.co_code, callee.co_consts)))
        elif isinstance(callee, type):
            parts.append(_type_version(callee, seen))
        elif _is_user_function(callee):
            parts.append(_code_version(callee, seen))
        else:
            parts.append(_opaque(callee))
    return hash_text("\0".join(parts))


def _type_version(cls: type, seen: set[int]) -> str:
    """Fingerprint a referenced class by its own methods, or opaque if not user code.

    A derivation that constructs a user-defined class and calls its methods depends on
    those method bodies; folding each one in (unwrapping ``classmethod`` /
    ``staticmethod`` and reading ``vars`` directly, so no descriptor runs) busts the
    cache when any of them change. Stdlib and third-party classes stay
    opaque-by-name-and-version. Guarded on ``seen`` so a method that refers back to its
    own class terminates.
    """
    if id(cls) in seen:  # cycle: a method references its own class
        return ""
    seen.add(id(cls))
    if not _is_user_class(cls):
        return _opaque(cls)
    parts = [_opaque(cls)]
    for name, member in sorted(vars(cls).items()):
        fn = getattr(member, "__func__", member)  # unwrap classmethod / staticmethod
        if isinstance(getattr(fn, "__code__", None), types.CodeType):
            parts.append(f"{name}={_code_version(fn, seen)}")
    return hash_text("\0".join(parts))


def _own_fingerprint(fn: Callable[..., Any]) -> str:
    """A reformatting-insensitive fingerprint of a function's own logic."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
        body = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (OSError, TypeError, SyntaxError):
        # REPL/exec/C: no source; fall back to the (less stable) code object.
        code = fn.__code__
        body = repr((code.co_code, code.co_consts, code.co_names))
    defaults = repr(
        (getattr(fn, "__defaults__", None), getattr(fn, "__kwdefaults__", None))
    )
    return f"{body}\0{defaults}"


def _referenced_namespaces(
    fn: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The module globals and the closure nonlocals a function references, by name.

    Returned separately because they are treated differently: global *values* are
    folded into the version, while nonlocals are not (a nonlocal may be mutated by the
    compute mid-run). Both, however, serve as roots for attribute and callee resolution.
    Both empty when ``fn`` is not a plain Python function.
    """
    try:
        closure_vars = inspect.getclosurevars(fn)
    except (TypeError, ValueError, OSError):  # not a plain Python function
        return {}, {}
    return dict(closure_vars.globals), dict(closure_vars.nonlocals)


def _referenced_callables(
    fn: Callable[..., Any], namespace: Mapping[str, Any]
) -> list[Any]:
    """Callables a function references via names, attributes, or nested code.

    Bare-name and closure callables come from ``namespace``; dotted references
    (``mod.func``, ``Cls.method``) are resolved by :func:`_attribute_callables`; and
    nested code objects (comprehensions, inner functions) come from ``co_consts``.
    """
    out: list[Any] = [v for v in namespace.values() if callable(v)]
    out += _attribute_callables(fn, namespace)
    code = getattr(fn, "__code__", None)
    if isinstance(code, types.CodeType):
        out += [c for c in code.co_consts if isinstance(c, types.CodeType)]
    return out


def _attribute_callables(
    fn: Callable[..., Any], namespace: Mapping[str, Any]
) -> list[Any]:
    """Callables reached through attribute access on a referenced module or class.

    Resolves each dotted reference in the source whose root name is a module or class in
    ``namespace``, so a call made through a module or class (``mod.func``), not a bare
    name, still contributes its callee. An attribute chain rooted at a runtime value
    (``self``, a parameter) is left unresolved: its type is not known statically.
    ``getattr`` runs only on modules and classes, so resolution cannot trigger an
    instance property or descriptor side effect.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):  # REPL / C / exec: no source to walk
        return []
    out: list[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            resolved = _resolve_attribute(node, namespace)
            if resolved is not None and callable(resolved):
                out.append(resolved)
    return out


def _resolve_attribute(node: ast.Attribute, namespace: Mapping[str, Any]) -> Any:
    """Resolve a dotted attribute chain to its object, traversing modules and classes.

    Returns ``None`` when the root name is unknown or when the chain would have to
    descend through a runtime instance, whose attributes cannot be resolved statically.
    """
    attrs: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    obj = namespace.get(current.id)
    for attr in reversed(attrs):
        if not isinstance(obj, types.ModuleType | type):
            return None  # only descend modules and classes; never getattr an instance
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    return obj


def _global_value_parts(namespace: Mapping[str, Any]) -> list[str]:
    """Fingerprints of the non-callable module globals a function reads, by name.

    A derivation that branches on a module-level constant depends on its value, so
    folding the value in busts the cache when the constant changes: the
    runtime-mutable-globals blind spot. Callables are covered by the callee walk and
    modules by attribute resolution, so both are skipped here.
    """
    parts: list[str] = []
    for name in sorted(namespace):
        value = namespace[name]
        if callable(value) or isinstance(value, types.ModuleType):
            continue
        parts.append(f"{name}={_value_fingerprint(value)}")
    return parts


def _value_fingerprint(value: Any) -> str:
    """A content fingerprint for a referenced global value, stable across processes."""
    return hash_text(_stable_repr(value))


def _stable_repr(value: Any) -> str:
    """A representation identical across processes for equal content.

    Scalars and nested containers render by content (dict and set orderings are
    normalized); anything else renders by its type identity, never by ``repr``: an
    address-bearing default ``repr`` would make the fingerprint differ run to run and
    defeat caching. A non-container object whose value should invalidate the cache is
    the residual the explicit ``code_version`` policy override covers.
    """
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(_stable_repr(v) for v in value) + "]"
    if isinstance(value, set | frozenset):
        return "{" + ",".join(sorted(_stable_repr(v) for v in value)) + "}"
    if isinstance(value, dict):
        items = sorted((repr(k), _stable_repr(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    return _opaque(type(value))


def _is_user_function(obj: Any) -> bool:
    """Whether ``obj`` is a user-defined function we should recurse into.

    Excludes builtins/C, stdlib, and installed third-party packages so the walk
    stays bounded to the project's own code.
    """
    if not isinstance(obj, types.FunctionType):
        return False
    filename = getattr(obj.__code__, "co_filename", "") or ""
    if not filename or filename.startswith("<"):  # <stdin>, <string>, exec
        return False
    if "site-packages" in filename or "dist-packages" in filename:
        return False
    top_module = (getattr(obj, "__module__", "") or "").split(".")[0]
    return top_module not in sys.stdlib_module_names


def _is_user_class(cls: Any) -> bool:
    """Whether ``cls`` is a class defined in the project's own code.

    Excludes builtins, stdlib, and installed third-party classes, so recursion into a
    class's methods stays bounded to project code (a foreign class stays opaque).
    """
    if not isinstance(cls, type):
        return False
    top_module = (getattr(cls, "__module__", "") or "").split(".")[0]
    if not top_module or top_module == "builtins":
        return False
    if top_module in sys.stdlib_module_names:
        return False
    try:
        filename = inspect.getfile(cls)
    except (TypeError, OSError):  # a dynamically built class has no source file
        return False
    return "site-packages" not in filename and "dist-packages" not in filename


def _opaque(obj: Any) -> str:
    """A stable identity string for a callable or type we don't recurse into.

    A third-party object is tagged with its distribution version, so a dependency
    upgrade that changes behavior busts the cache: the library-version-drift blind spot.
    Standard-library and builtin objects carry no distribution and stay a bare qualified
    name, since they move only with the interpreter version, which the fingerprint
    already includes.
    """
    module = getattr(obj, "__module__", "") or ""
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", repr(obj))
    name = f"{module}.{qualname}"
    version = _distribution_version(module)
    return f"{name}@{version}" if version else name


@cache
def _packages_distributions() -> Mapping[str, list[str]]:
    """Map each import name to its distributions (cached; the scan is slow)."""
    return importlib.metadata.packages_distributions()


@cache
def _distribution_version(module: str) -> str | None:
    """The installed version(s) of the distribution providing ``module``, or None.

    Maps the top-level import name to its distribution(s) and joins their versions.
    Returns None for the standard library, builtins, and any import with no installed
    distribution (a local module, a namespace stub), so only a genuine third-party
    dependency contributes a version.
    """
    top_module = module.split(".")[0] if module else ""
    if not top_module or top_module == "builtins":
        return None
    if top_module in sys.stdlib_module_names:
        return None
    distributions = _packages_distributions().get(top_module)
    if not distributions:
        return None
    versions: list[str] = []
    for dist in sorted(set(distributions)):
        try:
            versions.append(f"{dist}={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - defensive
            continue
    return ";".join(versions) or None


def source_version(source: str) -> str:
    """Return a code version for a derivation defined from source text.

    The counterpart to :func:`code_version` for agent-authored derivations: hashes
    the source's normalized AST plus the interpreter minor version. The source must
    parse.
    """
    py = f"py{sys.version_info.major}.{sys.version_info.minor}"
    body = ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)
    return hash_text("\0".join((body, py)))


def params_version(params: Mapping[str, Any]) -> str:
    """A version string for a set of parameter values."""
    return hash_json(dict(params))


def compose(*parts: str) -> str:
    """Combine component versions into one (order-independent for inputs)."""
    return hash_text("\n".join(parts))


def derivation_version(
    *,
    code: str,
    params: str,
    input_versions: Mapping[str, str],
) -> str:
    """Compose a derivation's data version from its code, params, and inputs.

    Inputs are folded in sorted by name, so the result is independent of mapping
    order.
    """
    ordered = [f"{name}={version}" for name, version in sorted(input_versions.items())]
    return compose(code, params, *ordered)
