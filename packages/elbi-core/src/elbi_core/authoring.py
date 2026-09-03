"""The agent-authoring loop: propose, verify, certify.

An agent proposes a derivation as generated source. It is registered with
``agent`` origin and ``proposed`` status, runs only under an isolating executor,
and is not served until certified. :func:`certify` is the local primitive that
advances the lifecycle; who may certify is a deployment concern.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from . import versioning
from .artifact import Artifact
from .derivation import Derivation, InputSpec, _refuse_in_process
from .errors import DerivationError, ElbiError
from .param import Param
from .quality.contract import DataContract
from .quality.verify import verify_contract
from .registry import Registry, active_registry
from .runner import Runner
from .serve import Serve
from .spec import validate_manifest


@dataclass(frozen=True)
class GoldenCase:
    """An expected input→output example a proposed derivation must reproduce.

    Running with ``params`` must produce ``expected``, compared by value (a content
    fingerprint), so any correct implementation passes. Set ``unordered`` to
    compare list-of-row output as a multiset, for results whose row order is not
    meaningful.
    """

    expected: Any
    params: Mapping[str, Any] = field(default_factory=dict)
    unordered: bool = False
    name: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of running a proposed derivation, before certifying.

    ``ok`` is true when the derivation ran without error, every supplied golden case
    passed, *and* (when the derivation declares a ``claim`` or a ``contract``) the
    verification oracle finds its conclusion sound and the contract checker finds its
    output valid. ``matched_prediction`` is whether all golden cases passed, or ``None``
    when none were given. ``cases_total`` and ``cases_passed`` report how many golden
    cases ran and passed. ``oracle_verdict`` is the oracle's verdict on the declared
    conclusion (``None`` when no claim was declared); ``contract_verdict`` is the
    checker's verdict on the declared contract (``None`` when none was declared).
    """

    ok: bool
    artifact: Artifact | None = None
    rendered: str | None = None
    matched_prediction: bool | None = None
    error: str | None = None
    cases_total: int = 0
    cases_passed: int = 0
    oracle_verdict: str | None = None
    oracle_detail: str | None = None
    oracle_attestation: dict[str, Any] | None = None
    contract_verdict: str | None = None
    contract_detail: str | None = None
    contract_attestation: dict[str, Any] | None = None
    #: The derivation's content-addressed data version (a hash of its code, params, and
    #: input versions) when it ran. The run's input identity: an identical re-run yields
    #: the same version, and a change to the code or data a new one. Surfaced so a
    #: tracking layer can key a certified run on it.
    data_version: str | None = None
    #: The components the data version composes: the code's own version and each input's
    #: version by key. Kept alongside the composite so a result history can attribute a
    #: moved estimate to the code versus the data, not merely record that it moved.
    code_version: str | None = None
    input_versions: Mapping[str, str] | None = None


def propose(
    name: str,
    source: str,
    *,
    serve: Serve | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    params: Mapping[str, Param] | None = None,
    description: str | None = None,
    claim: Mapping[str, Any] | None = None,
    contract: DataContract | None = None,
    deps: Sequence[str] | None = None,
    registry: Registry | None = None,
) -> Derivation:
    """Register agent-generated ``source`` as a proposed, agent-origin derivation.

    The source MUST parse and MUST define a top-level function named ``name``;
    that function is the compute, run later under a sandbox executor. The
    derivation is registered as ``proposed`` and is not served until certified.
    ``claim`` declares the conclusion the derivation asserts as verification column
    roles (e.g. ``{"x": "education", "y": "income"}``); when given, certification
    runs the oracle on the output and certifies only a sound conclusion. ``contract``
    declares a data contract the output must satisfy; when given, certification checks
    the output against it and certifies only a sound (valid) result. ``deps``
    names third-party packages the source imports (e.g. ``["numpy", "scikit-learn"]``);
    they are provisioned into the sandbox when the derivation runs.

    Raises:
        DerivationError: if the source does not parse or does not define ``name``.
        SpecValidationError: if the resulting manifest is not spec-conformant.
    """
    target = registry if registry is not None else active_registry()
    _check_source(name, source)
    if contract is not None:
        contract.validate()

    resolved_inputs: Mapping[str, InputSpec] = dict(inputs or {})
    depends_on = tuple(
        dict.fromkeys(
            dep.name for dep in resolved_inputs.values() if isinstance(dep, Derivation)
        )
    )
    derivation = Derivation(
        name=name,
        compute=_refuse_in_process,
        serve=serve,
        inputs=resolved_inputs,
        params=dict(params or {}),
        depends_on=depends_on,
        description=(
            description if description is not None else _docstring(source, name)
        ),
        origin="agent",
        status="proposed",
        source=source,
        claim=dict(claim) if claim else None,
        contract=contract,
        deps=tuple(deps or ()),
    )
    validate_manifest(derivation.to_manifest())
    target.register(derivation)
    return derivation


def verify(
    runner: Runner,
    derivation: Derivation,
    *,
    predicted: Any | None = None,
    cases: Sequence[GoldenCase] | None = None,
    check_determinism: bool = True,
) -> VerificationResult:
    """Run ``derivation`` and report whether it is fit to certify.

    The ``runner`` should use a sandbox executor, since agent-authored derivations
    refuse to run in-process. ``cases`` checks the output against expected examples
    (computation: did it produce the right value); ``predicted`` is the one-case
    shorthand. When the derivation declares a ``claim``, the verification oracle
    also checks that its *conclusion* is sound: the inferential check golden
    cases cannot provide. When ``check_determinism`` is set (the default), the
    derivation is also run a second time and its output compared, so a result that is
    not reproducible (an unseeded model, a clock read) cannot certify. When the
    derivation declares a ``contract``, the output is checked against it as well,
    resolving referential-integrity targets through ``runner``. The result is ``ok``
    only if every golden case passes, the output is reproducible, *and* (when declared)
    both the oracle finds the conclusion sound and the contract holds. Execution
    failures are captured, not raised.
    """
    effective = _effective_cases(predicted, cases)
    try:
        if not effective:
            artifact = runner.run(derivation.name)
            reason = (
                _determinism_error(runner, derivation.name, {}, artifact)
                if check_determinism
                else None
            )
            if reason is not None:
                return VerificationResult(
                    ok=False,
                    artifact=artifact,
                    rendered=_render(derivation, artifact),
                    error=reason,
                )
            verdict, detail, attestation = _oracle_check(derivation, artifact)
            cverdict, cdetail, cattest = _contract_check(runner, derivation, artifact)
            version, code_version, input_versions = _version_parts(
                runner, derivation.name, {}
            )
            return VerificationResult(
                ok=_sound(verdict) and _sound(cverdict),
                artifact=artifact,
                rendered=_render(derivation, artifact),
                matched_prediction=None,
                oracle_verdict=verdict,
                oracle_detail=detail,
                oracle_attestation=attestation,
                contract_verdict=cverdict,
                contract_detail=cdetail,
                contract_attestation=cattest,
                data_version=version,
                code_version=code_version,
                input_versions=input_versions,
            )
        outputs: list[Artifact] = []
        passed = 0
        for case in effective:
            artifact = runner.run(derivation.name, dict(case.params))
            outputs.append(artifact)
            if _case_passes(artifact, case):
                passed += 1
        first = outputs[0]  # effective is non-empty, so at least one case ran
        params = dict(effective[0].params)
        reason = (
            _determinism_error(runner, derivation.name, params, first)
            if check_determinism
            else None
        )
    except DerivationError as exc:
        return VerificationResult(ok=False, error=str(exc))

    all_passed = passed == len(effective)
    if reason is not None:
        return VerificationResult(
            ok=False,
            artifact=first,
            rendered=_render(derivation, first),
            matched_prediction=all_passed,
            cases_total=len(effective),
            cases_passed=passed,
            error=reason,
        )
    verdict, detail, attestation = _oracle_check(derivation, first)
    cverdict, cdetail, cattest = _contract_check(runner, derivation, first)
    version, code_version, input_versions = _version_parts(
        runner, derivation.name, params
    )
    return VerificationResult(
        ok=all_passed and _sound(verdict) and _sound(cverdict),
        artifact=first,
        rendered=_render(derivation, first),
        matched_prediction=all_passed,
        cases_total=len(effective),
        cases_passed=passed,
        oracle_verdict=verdict,
        oracle_detail=detail,
        oracle_attestation=attestation,
        contract_verdict=cverdict,
        contract_detail=cdetail,
        contract_attestation=cattest,
        data_version=version,
        code_version=code_version,
        input_versions=input_versions,
    )


def _sound(verdict: str | None) -> bool:
    """Whether a gate's verdict permits certification (undeclared or sound)."""
    return verdict is None or verdict == "sound"


def _version_parts(
    runner: Runner, name: str, params: Mapping[str, Any]
) -> tuple[str | None, str | None, Mapping[str, str] | None]:
    """The data version and its components for a run, or Nones if uncomputable.

    Fingerprints the derivation's code, params, and input versions without recomputing
    the result, returning ``(data_version, code_version, input_versions)``. Best-effort:
    a fingerprinting failure must not fail a verify that already ran, so it degrades to
    Nones rather than raising.
    """
    try:
        version, code, inputs = runner.version_parts(name, dict(params))
    except ElbiError:  # pragma: no cover - inputs already resolved for the run
        return None, None, None
    return version, code, inputs


def _determinism_error(
    runner: Runner, name: str, params: Mapping[str, Any], first: Artifact
) -> str | None:
    """Re-run ``name`` once and return a reason if its output is not reproducible.

    A certified result must be a pure function of its declared inputs. The re-run forces
    a fresh execution (bypassing the cache), and the comparison tolerates tiny
    floating-point differences (multithreaded BLAS reductions are not bit-reproducible
    even for a model fit with the same seed), while catching gross nondeterminism: a
    different random seed, a clock or process-id read, or an unstable row order.
    """
    second = runner.run(name, dict(params), fresh=True)
    if _reproducible(first.value, second.value):
        return None
    return (
        "not reproducible: the derivation produced different output on two identical "
        "runs, so a certified result would not be a pure function of its inputs; seed "
        "any random number generator and do not depend on the clock, the process id, "
        "or unordered iteration order"
    )


def _reproducible(a: Any, b: Any, *, rtol: float = 1e-4, atol: float = 1e-8) -> bool:
    """Whether two outputs agree up to floating-point tolerance.

    Numbers compare with a relative tolerance so a nondeterministic parallel reduction
    (the same model, same seed, run twice) does not read as nondeterminism, while a
    different seed, an embedded timestamp, or a reordered result, all far larger than
    the tolerance, does. Booleans compare by identity so ``True`` never matches ``1``.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _reproducible(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_reproducible(a[k], b[k]) for k in a)
    return bool(a == b)


def _oracle_check(
    derivation: Derivation, artifact: Artifact
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Run the verification oracle on a derivation's declared conclusion.

    Returns ``(None, None, None)`` when no claim is declared. Otherwise runs the
    oracle's composite check over the output rows with the claim's column roles and
    returns its verdict (``sound`` / ``unsound`` / ``invalid`` / ``inconclusive``), a
    one-line detail, and a self-describing attestation record to ride with the served
    result. An output that is not row data cannot support a conclusion, so it is
    reported ``inconclusive``, which, like ``unsound``, blocks certification.
    """
    if not derivation.claim:
        return None, None, None
    from .verification import verify_all

    value = artifact.value
    if not (isinstance(value, list) and all(isinstance(r, dict) for r in value)):
        return "inconclusive", "claim cannot be checked: output is not row data", None
    report = verify_all(value, **dict(derivation.claim))
    detail = report.ran[0].detail if report.ran else "no applicable check"
    attestation = {
        "schema": "elbi.verification/v1",
        "verdict": report.verdict,
        "claim": dict(derivation.claim),
        # The magnitude the oracle itself computed and a unit-aware description of it,
        # so a served answer reports the certified number rather than one the model
        # wrote; ``adjusted_for`` records what it holds fixed (direct-effect controls).
        "estimate": report.estimate,
        "estimate_label": report.estimate_label,
        "adjusted_for": list(report.adjusted_for),
        "checks": [
            {"name": g.name, "verdict": g.verdict, "detail": g.detail}
            for g in report.ran
        ],
        "skipped": [name for name, _ in report.skipped],
        "data_hash": versioning.hash_json(value),
    }
    return report.verdict, detail, attestation


def _contract_check(
    runner: Runner, derivation: Derivation, artifact: Artifact
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Check a derivation's declared data contract against its output.

    Returns ``(None, None, None)`` when no contract is declared. Otherwise checks the
    output rows against the contract, resolving each foreign key's referenced resource
    through the runner, and returns the verdict (``sound`` / ``unsound`` /
    ``inconclusive``), a one-line detail, and a self-describing attestation to ride with
    the served result. Output that is not row data cannot satisfy a contract, so it is
    reported ``inconclusive``, which, like ``unsound``, blocks certification.
    """
    contract = derivation.contract
    if contract is None:
        return None, None, None
    value = artifact.value
    if not (isinstance(value, list) and all(isinstance(r, dict) for r in value)):
        return (
            "inconclusive",
            "contract cannot be checked: output is not row data",
            None,
        )
    references = _resolve_references(runner, contract)
    report = verify_contract(value, contract, references=references)
    attestation = {**report.attestation(), "data_hash": versioning.hash_json(value)}
    return report.verdict, _contract_detail(report), attestation


def _contract_detail(report: Any) -> str:
    """A one-line summary of a contract report: the first failing clause, or a pass."""
    for clause in report.clauses:
        if clause.verdict != "sound":
            return f"{clause.name}: {clause.detail}"
    return f"{len(report.clauses)} clauses hold over {report.row_count} rows"


def _resolve_references(
    runner: Runner, contract: DataContract
) -> dict[str, list[dict[str, Any]]]:
    """Best-effort resolution of a contract's foreign-key targets to their rows.

    Each referenced resource is loaded as an upstream derivation's output if one is
    registered under that name, else as a bound dataset. A resource that cannot be
    resolved is omitted, so its foreign-key clause reports ``inconclusive`` rather than
    a false pass.
    """
    references: dict[str, list[dict[str, Any]]] = {}
    for fk in contract.table.foreign_keys:
        resource = fk.reference.resource
        if resource in references:
            continue
        rows = _load_resource(runner, resource)
        if rows is not None:
            references[resource] = rows
    return references


def _load_resource(runner: Runner, resource: str) -> list[dict[str, Any]] | None:
    """Load a resource's rows as a derivation output or a bound dataset, or None."""
    try:
        value = runner.run(resource).value
        if isinstance(value, list) and all(isinstance(r, dict) for r in value):
            return value
    except ElbiError:
        pass
    try:
        return runner.dataset(resource).rows
    except ElbiError:
        return None


def certify(derivation: Derivation, *, registry: Registry | None = None) -> Derivation:
    """Advance ``derivation`` to ``certified`` and re-register it.

    Returns the new certified value; derivations are immutable.
    """
    target = registry if registry is not None else active_registry()
    certified = replace(derivation, status="certified")
    target.replace(certified)
    return certified


@runtime_checkable
class CertificationPolicy(Protocol):
    """Decides whether a verified proposal is certified now or held for review.

    Pluggable: the default auto-certifies anything that passes verification; a
    governed deployment can hold proposals for an explicit human :func:`certify`.
    """

    def should_certify(
        self, derivation: Derivation, result: VerificationResult
    ) -> bool:
        """Return whether ``derivation`` should be certified given its ``result``."""
        ...


class AutoCertifyOnVerify:
    """Certify automatically when verification passes. The default policy.

    Provenance (``origin=agent``) is kept, so the certification stays auditable
    and reversible.
    """

    def should_certify(
        self, derivation: Derivation, result: VerificationResult
    ) -> bool:
        """Certify iff verification succeeded."""
        return result.ok


class ManualCertification:
    """Hold every proposal for an explicit human :func:`certify`.

    For deployments where a served answer must be human-endorsed.
    """

    def should_certify(
        self, derivation: Derivation, result: VerificationResult
    ) -> bool:
        """Never certify automatically."""
        return False


class RequireChecks:
    """Auto-certify only when the proposal passed at least one golden case.

    Certifies without a human, but only on execution-grounded evidence, never on a
    run with nothing to check against.
    """

    def should_certify(
        self, derivation: Derivation, result: VerificationResult
    ) -> bool:
        """Certify iff verification passed and at least one case was checked."""
        return result.ok and result.cases_total > 0


class RequireSoundConclusion:
    """Auto-certify only a derivation that declares a claim the oracle finds sound.

    The strictest policy: certifies without a human, but only on a derivation that
    states the conclusion it asserts *and* whose conclusion the verification oracle
    confirms (verdict ``sound``). For governed deployments where every served
    derivation must carry a verified inference, not merely a correct computation.
    """

    def should_certify(
        self, derivation: Derivation, result: VerificationResult
    ) -> bool:
        """Certify iff verification passed and the oracle confirmed the conclusion."""
        return result.ok and result.oracle_verdict == "sound"


class RequireSoundContract:
    """Auto-certify only a derivation that declares a contract the checker finds sound.

    The data-cleaning counterpart to :class:`RequireSoundConclusion`: certifies without
    a human, but only when the derivation states the contract its output must satisfy
    *and* the checker confirms it (verdict ``sound``). For governed deployments where a
    served table must carry a verified quality bar, not merely a correct computation.
    """

    def should_certify(
        self, derivation: Derivation, result: VerificationResult
    ) -> bool:
        """Certify iff verification passed and the checker confirmed the contract."""
        return result.ok and result.contract_verdict == "sound"


#: The default certification policy: certify on a clean verify (which, when a claim is
#: declared, already requires the oracle to find the conclusion sound).
DEFAULT_CERTIFICATION: CertificationPolicy = AutoCertifyOnVerify()


@dataclass(frozen=True)
class AuthorOutcome:
    """The result of the full author loop.

    Carries the (possibly certified) derivation, its verification result, and
    whether the policy certified it.
    """

    derivation: Derivation
    result: VerificationResult
    certified: bool


def author(
    name: str,
    source: str,
    *,
    runner: Runner,
    serve: Serve | None = None,
    inputs: Mapping[str, InputSpec] | None = None,
    params: Mapping[str, Param] | None = None,
    description: str | None = None,
    claim: Mapping[str, Any] | None = None,
    contract: DataContract | None = None,
    deps: Sequence[str] | None = None,
    predicted: Any | None = None,
    cases: Sequence[GoldenCase] | None = None,
    check_determinism: bool = True,
    policy: CertificationPolicy | None = None,
    registry: Registry | None = None,
) -> AuthorOutcome:
    """Run the whole loop: propose, verify, then certify per ``policy``.

    Verifies against any ``cases``/``predicted`` (computation), the reproducibility of
    the output (unless ``check_determinism`` is cleared), and, when declared, the
    verification oracle (conclusion soundness) and the data contract (output validity),
    then consults ``policy`` (default :data:`DEFAULT_CERTIFICATION`). A proposal that
    fails verification (a wrong output, a result that does not reproduce, an unsound
    declared conclusion, *or* an output that breaks its contract) is never certified and
    stays ``proposed``.
    """
    target = registry if registry is not None else active_registry()
    decision = policy if policy is not None else DEFAULT_CERTIFICATION

    derivation = propose(
        name,
        source,
        serve=serve,
        inputs=inputs,
        params=params,
        description=description,
        claim=claim,
        contract=contract,
        deps=deps,
        registry=target,
    )
    result = verify(
        runner,
        derivation,
        predicted=predicted,
        cases=cases,
        check_determinism=check_determinism,
    )
    certified = decision.should_certify(derivation, result)
    if certified:
        derivation = certify(derivation, registry=target)
    return AuthorOutcome(derivation=derivation, result=result, certified=certified)


def _effective_cases(
    predicted: Any | None, cases: Sequence[GoldenCase] | None
) -> list[GoldenCase]:
    """Normalize the two ways of stating expectations into one case list.

    ``cases`` wins when given; otherwise a bare ``predicted`` becomes a single
    no-params case. Passing neither yields an empty list (a run-only verify).
    """
    if cases is not None:
        return list(cases)
    if predicted is not None:
        return [GoldenCase(expected=predicted)]
    return []


def _case_passes(artifact: Artifact, case: GoldenCase) -> bool:
    """Whether ``artifact``'s value matches the case's expected value."""
    return _fingerprint(artifact.value, unordered=case.unordered) == _fingerprint(
        case.expected, unordered=case.unordered
    )


def _fingerprint(value: Any, *, unordered: bool) -> str:
    """A content fingerprint for output comparison.

    Column order never matters (the JSON hash sorts keys). Row order matters only
    when ``unordered`` is false: for an unordered list of rows, the per-row hashes
    are sorted first, so the same rows in a different order fingerprint the same.
    """
    if (
        unordered
        and isinstance(value, list)
        and all(isinstance(r, dict) for r in value)
    ):
        return versioning.hash_json(sorted(versioning.hash_json(row) for row in value))
    return versioning.hash_json(value)


def _render(derivation: Derivation, artifact: Artifact) -> str | None:
    """Render an artifact through the serve contract, if the derivation serves."""
    if derivation.serve is None:
        return None
    return derivation.serve.render(artifact)


def _check_source(name: str, source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise DerivationError(
            f"proposed source for {name!r} is not valid Python: {exc}"
        ) from exc
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    if name not in defined:
        raise DerivationError(
            f"proposed source must define a top-level function named {name!r}"
        )


def _docstring(source: str, name: str) -> str | None:
    """The first line of the named function's docstring, if any."""
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name == name:
            doc = ast.get_docstring(node)
            return doc.strip().splitlines()[0] if doc else None
    return None  # pragma: no cover - _check_source guarantees the function exists
