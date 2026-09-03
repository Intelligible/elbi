"""The enforcement is proven at the loop boundary with a scripted LLM.

A verified answer can only come from ``derive``: the model authors its analysis as a
derivation and the oracle gates the output. Here the injected derive fn runs the *real*
oracle on the dataset (standing in for authoring the derivation), so each test asserts
the loop's guarantee against genuine verdicts: a real effect verifies only after passing
the gate, an unsound attempt is rejected and the loop continues, noise never becomes a
verified effect, a free-text answer is not trusted, and the step budget terminates.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from elbi_agent import (
    DeriveOutcome,
    Step,
    ToolCall,
    ToolSpec,
    Transcript,
    Workspace,
    answer,
)
from elbi_core import verify_all
from elbi_core.versioning import hash_json

N = 600

# The compute is irrelevant to these loop tests; the injected derive fn runs the oracle
# on the fixed dataset, so a trivial source stands in for the authored derivation.
_SRC = "def d(ctx):\n    return ctx.input('d')"


def _rows(**columns: list[float]) -> list[dict[str, str]]:
    names = list(columns)
    return [{c: str(columns[c][i]) for c in names} for i in range(N)]


def _deriving(rows: list[dict[str, str]]):
    """A derive fn that authors nothing but runs the real oracle on ``rows``.

    This keeps enforcement genuine at the loop boundary: the claim the model declares is
    checked by the same oracle the real authoring path uses, so a sound result is a real
    one and an unsound claim cannot slip through.
    """

    def derive(
        name: str,
        source: str,
        claim: Mapping[str, Any] | None,
        contract: Mapping[str, Any] | None,
        fmt: str,
        assumptions: Sequence[str],
        deps: Sequence[str],
    ) -> DeriveOutcome:
        report = verify_all(rows, **dict(claim or {}))
        checks = tuple((g.name, g.verdict, g.detail) for g in report.ran)
        if report.verdict == "sound":
            return DeriveOutcome(
                True,
                "sound",
                "output rows",
                checks,
                hash_json(rows),
                rows=tuple(rows),
                estimate=report.estimate,
                estimate_label=report.estimate_label,
                adjusted_for=report.adjusted_for,
            )
        return DeriveOutcome(
            False, report.verdict, "output rows", checks, detail=report.render()
        )

    return derive


def _workspace(rows: list[dict[str, str]]) -> Workspace:
    return Workspace(datasets={"d": rows}, derive=_deriving(rows))


def _signflip(seed: int = 0) -> list[dict[str, str]]:
    # A Simpson's-paradox reveal: marginal x~y is negative, but holding z fixed the
    # coefficient is +0.6. The sign turns entirely on whether z is controlled, and from
    # data alone z's role (a confounder to adjust for, or a collider/mediator not to) is
    # unidentifiable, so the controlled effect is not certifiable either way.
    rng = random.Random(seed)
    z = [rng.gauss(0, 1) for _ in range(N)]
    x = [zi + rng.gauss(0, 0.5) for zi in z]
    y = [0.6 * xi - 1.5 * zi + rng.gauss(0, 0.5) for xi, zi in zip(x, z)]
    return _rows(x=x, y=y, z=z)


def _robust(seed: int = 0) -> list[dict[str, str]]:
    # A genuinely certifiable effect: x raises y by ~0.6, and w is a confounder that
    # refines the estimate without reversing it: dropping w leaves the effect intact, so
    # it survives the gate and is verified sound.
    rng = random.Random(seed)
    w = [rng.gauss(0, 1) for _ in range(N)]
    x = [0.5 * wi + rng.gauss(0, 1) for wi in w]
    y = [0.6 * xi + 0.4 * wi + rng.gauss(0, 0.5) for xi, wi in zip(x, w)]
    return _rows(x=x, y=y, w=w)


def _noise(seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return _rows(
        x=[rng.gauss(0, 1) for _ in range(N)],
        y=[rng.gauss(0, 1) for _ in range(N)],
    )


class _Scripted:
    """An LLMClient that replays a fixed list of steps, ignoring the transcript."""

    def __init__(self, *steps: Step) -> None:
        self._steps = list(steps)
        self._i = 0

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        step = self._steps[self._i]
        self._i += 1
        return step


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=dict(arguments))


def _derive(deriv_name: str, **claim: object) -> ToolCall:
    return ToolCall(
        id="c-derive",
        name="derive",
        arguments={"name": deriv_name, "source": _SRC, "claim": dict(claim)},
    )


def test_effect_verified_when_robust_to_controls() -> None:
    # A real effect that survives its adjustment set verifies through the oracle. The
    # model's own prose leads the answer, and the system appends a certified-findings
    # block carrying the oracle's OWN estimate, so the reported magnitude is the one the
    # oracle computed, not a number the model typed.
    workspace = _workspace(_robust())
    client = _Scripted(
        Step(tool_calls=(_call("structure_map", dataset="d"),)),
        Step(tool_calls=(_derive("x_on_y", x="x", y="y", controls=["w"]),)),
        Step(
            tool_calls=(_call("answer", summary="x raises y by about 0.6 per unit."),)
        ),
    )
    result = answer("What is the effect of x on y?", workspace, client)
    assert result.verified and result.verdict == "sound"
    # the model's words lead, then the system-rendered certified block is appended
    assert result.narrative.startswith("x raises y by about 0.6 per unit.")
    assert "Certified findings" in result.narrative
    # the certified magnitude in the block is the oracle's, and labelled as adjusted
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.estimate is not None and finding.estimate_label is not None
    assert finding.estimate_label in result.narrative  # the oracle's number, rendered
    assert finding.adjusted_for == ("w",)
    assert result.spec is not None and result.spec["derivation"] == "x_on_y"
    assert result.spec["claim"]["controls"] == ["w"]
    assert result.data_hash is not None and len(result.data_hash) == 64
    assert any(name == "effect" for name, _, _ in result.checks)
    assert result.steps == 3


def test_multiple_variables_accumulate_as_certified_findings() -> None:
    # "How does each variable affect y" authors one derivation per variable. Every
    # certified effect must survive into the answer with the oracle's own estimate, not
    # collapse to the last derivation, and not be restated from the model's prose.
    # (Reverting the accumulation to overwrite-the-last drops x_on_y, failing this.)
    workspace = _workspace(_robust())  # x, y, w; both x->y and w->y are real
    client = _Scripted(
        Step(tool_calls=(_derive("x_effect", x="x", y="y", controls=["w"]),)),
        Step(tool_calls=(_derive("w_effect", x="w", y="y", controls=["x"]),)),
        Step(tool_calls=(_call("answer", summary="Both x and w raise y."),)),
    )
    result = answer("How does each variable affect y?", workspace, client)
    assert result.verified and result.verdict == "sound"
    assert {f.name for f in result.findings} == {"x_effect", "w_effect"}
    # each finding carries the oracle's own estimate, and both are rendered in the block
    assert all(f.estimate is not None for f in result.findings)
    for f in result.findings:
        assert f.estimate_label is not None and f.estimate_label in result.narrative
    assert result.narrative.startswith("Both x and w raise y.")


def test_effect_contingent_on_a_control_is_not_certified() -> None:
    # The Simpson reveal: the effect exists only after controlling z, whose causal
    # role the data cannot settle. The gate refuses to certify it, so the model cannot
    # fake a verified answer and must concede.
    workspace = _workspace(_signflip())
    client = _Scripted(
        Step(tool_calls=(_derive("x_on_y", x="x", y="y"),)),
        Step(tool_calls=(_derive("x_on_y_adj", x="x", y="y", controls=["z"]),)),
        Step(tool_calls=(_call("conclude_inconclusive", reason="z role unknown"),)),
    )
    result = answer("What is the effect of x on y?", workspace, client)
    assert not result.verified and result.verdict == "inconclusive"


def test_noise_cannot_become_a_verified_effect() -> None:
    workspace = _workspace(_noise())
    client = _Scripted(
        Step(tool_calls=(_derive("x_on_y", x="x", y="y"),)),
        Step(
            tool_calls=(_call("conclude_inconclusive", reason="no signal in the data"),)
        ),
    )
    result = answer("What is the effect of x on y?", workspace, client)
    assert not result.verified and result.verdict == "inconclusive"


def test_prose_answer_is_not_trusted_as_verified() -> None:
    workspace = _workspace(_signflip())
    client = _Scripted(Step(text="x raises y.", end=True))
    result = answer("What is the effect of x on y?", workspace, client)
    assert not result.verified and result.verdict == "unverified"


def test_step_budget_terminates_unverified() -> None:
    workspace = _workspace(_signflip())
    never_terminal = [
        Step(tool_calls=(_call("describe_dataset", dataset="d"),)) for _ in range(20)
    ]
    client = _Scripted(*never_terminal)
    result = answer("What is the effect of x on y?", workspace, client, max_steps=3)
    assert not result.verified and "budget" in result.narrative.lower()
    assert result.steps == 3


def test_budget_degrades_to_a_grounded_conclusion() -> None:
    # Hitting the budget hands the model one final turn to conclude with what it
    # explored, so a hard question ends in the best available answer, not a dead stop.
    workspace = _workspace(_two_groups())
    explore = [
        Step(tool_calls=(_call("run_code", code="result = 1"),)) for _ in range(3)
    ]
    closing = Step(tool_calls=(_call("answer", summary="here is what I found"),))
    client = _Scripted(*explore, closing)
    result = answer("a hard question?", workspace, client, max_steps=3)
    assert result.verdict == "grounded"
    assert result.narrative == "here is what I found"


def _two_groups() -> list[dict[str, str]]:
    rng = random.Random(1)
    rows: list[dict[str, str]] = []
    for _ in range(N):
        rows.append({"g": "a", "v": str(rng.gauss(1.0, 1.0))})
        rows.append({"g": "b", "v": str(rng.gauss(0.0, 1.0))})
    return rows


def test_comparison_gate_routes_and_verifies() -> None:
    # A non-effect question routes through the same derive gate, to comparison.
    workspace = _workspace(_two_groups())
    client = _Scripted(
        Step(tool_calls=(_derive("group_diff", group="g", value="v"),)),
        Step(tool_calls=(_call("answer", summary="group a scores higher than b."),)),
    )
    result = answer("Do groups a and b differ on v?", workspace, client)
    assert result.verified and result.verdict == "sound"
    assert any(name == "comparison" for name, _, _ in result.checks)


def _predictive() -> list[dict[str, str]]:
    rng = random.Random(2)
    rows = []
    for _ in range(N):
        f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
        y = 1 if (0.9 * f1 + 0.5 * f2 + rng.gauss(0, 1)) > 0 else 0
        rows.append({"f1": str(f1), "f2": str(f2), "y": str(y)})
    return rows


def test_prediction_gate_routes_and_verifies() -> None:
    # a predictiveness claim is answered through its own gate, re-evaluated honestly
    workspace = _workspace(_predictive())
    client = _Scripted(
        Step(tool_calls=(_derive("y_from_f", features=["f1", "f2"], target="y"),)),
        Step(
            tool_calls=(_call("answer", summary="f1 and f2 predict y with AUC ~0.8."),)
        ),
    )
    result = answer("How well can f1 and f2 predict y?", workspace, client)
    assert result.verified and result.verdict == "sound"
    assert any(name == "prediction" for name, _, _ in result.checks)


def _clean_ab() -> list[dict[str, str]]:
    rng = random.Random(3)
    rows = []
    for _ in range(4000):
        v = "A" if rng.random() < 0.5 else "B"
        conv = 1 if rng.random() < (0.20 if v == "A" else 0.30) else 0
        rows.append({"variant": v, "converted": str(conv)})
    return rows


def test_experiment_gate_routes_and_verifies() -> None:
    # an A/B claim leaves the loop only through its own gate (SRM + metric)
    workspace = _workspace(_clean_ab())
    client = _Scripted(
        Step(tool_calls=(_derive("ab_lift", variant="variant", metric="converted"),)),
        Step(
            tool_calls=(
                _call("answer", summary="variant B lifted conversion by ~10pp."),
            )
        ),
    )
    result = answer("Did variant B lift conversion?", workspace, client)
    assert result.verified and result.verdict == "sound"
    assert any(name == "experiment" for name, _, _ in result.checks)


def _blobs() -> list[dict[str, str]]:
    rng = random.Random(4)
    rows = []
    for _ in range(500):
        cx, cy = rng.choice([(-6, -6), (6, 6), (-6, 6)])
        rows.append(
            {
                "f0": str(round(cx + rng.gauss(0, 1), 4)),
                "f1": str(round(cy + rng.gauss(0, 1), 4)),
            }
        )
    return rows


def test_clusters_gate_routes_and_verifies() -> None:
    workspace = _workspace(_blobs())
    client = _Scripted(
        Step(tool_calls=(_derive("segments", features=["f0", "f1"], k=3),)),
        Step(tool_calls=(_call("answer", summary="there are three real segments."),)),
    )
    result = answer("Are there real customer segments?", workspace, client)
    assert result.verified and result.verdict == "sound"
    assert any(name == "clusters" for name, _, _ in result.checks)


def test_answer_is_grounded_only_after_running_code() -> None:
    workspace = _workspace(_two_groups())
    # answer after run_code -> grounded
    grounded = _Scripted(
        Step(tool_calls=(_call("run_code", code="result = len(data['d'])"),)),
        Step(tool_calls=(_call("answer", summary="there are many rows"),)),
    )
    result = answer("How many rows are there?", workspace, grounded)
    assert not result.verified and result.verdict == "grounded"
    # answer with no execution behind it -> unverified, not trusted
    ungrounded = _Scripted(Step(tool_calls=(_call("answer", summary="about 1400"),)))
    result = answer("How many rows are there?", workspace, ungrounded)
    assert result.verdict == "unverified"


def test_declared_assumptions_ride_with_the_verified_answer() -> None:
    # A context-supplied identification assumption is recorded as an audited premise on
    # the verified result, never a silent change to the oracle's row-based verdict.
    workspace = _workspace(_robust())
    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-derive",
                    name="derive",
                    arguments={
                        "name": "x_on_y",
                        "source": _SRC,
                        "claim": {"x": "x", "y": "y", "controls": ["w"]},
                        "assumptions": ["treatment was randomized (spec §2)"],
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="x causes y, given randomization."),)),
    )
    result = answer("Does x cause y?", workspace, client)
    assert result.verified and result.verdict == "sound"
    assert result.assumptions == ("treatment was randomized (spec §2)",)


def test_context_is_accepted_and_does_not_break_the_run() -> None:
    # Context is injected for the model to read; the run still verifies from the rows.
    workspace = _workspace(_robust())
    client = _Scripted(
        Step(tool_calls=(_derive("x_on_y", x="x", y="y", controls=["w"]),)),
        Step(tool_calls=(_call("answer", summary="x raises y."),)),
    )
    result = answer(
        "effect of x on y?",
        workspace,
        client,
        context="x = the exposure; y = the outcome; w = a confounder.",
    )
    assert result.verified and result.verdict == "sound"


def test_derive_unavailable_without_capability() -> None:
    # A workspace with no derive capability cannot reach a verified answer.
    workspace = Workspace(datasets={"d": _robust()})
    client = _Scripted(
        Step(tool_calls=(_derive("x_on_y", x="x", y="y"),)),
        Step(tool_calls=(_call("conclude_inconclusive", reason="cannot derive here"),)),
    )
    result = answer("What is the effect of x on y?", workspace, client)
    assert not result.verified


def test_derive_forwards_declared_deps_to_the_derive_fn() -> None:
    # The model declares the packages its derivation source imports; the loop must hand
    # them to the derive fn so the sandbox provisions them (else numpy/sklearn imports
    # fail and no real model can ever be certified).
    seen: dict[str, Sequence[str]] = {}

    def recording_derive(
        name: str,
        source: str,
        claim: Mapping[str, Any] | None,
        contract: Mapping[str, Any] | None,
        fmt: str,
        assumptions: Sequence[str],
        deps: Sequence[str],
    ) -> DeriveOutcome:
        seen["deps"] = tuple(deps)
        return DeriveOutcome(True, "sound", "ok", (), hash_json([{"v": 1}]))

    workspace = Workspace(datasets={"d": _robust()}, derive=recording_derive)
    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-derive",
                    name="derive",
                    arguments={
                        "name": "m",
                        "source": _SRC,
                        "claim": {"x": "x", "y": "y"},
                        "deps": ["numpy", "scikit-learn"],
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="done"),)),
    )
    answer("effect of x on y?", workspace, client)
    assert seen["deps"] == ("numpy", "scikit-learn")


def test_run_code_persists_across_calls_with_a_scratch_dir(tmp_path: Path) -> None:
    # With a scratch dir, a file one run_code call writes is there for the next, so the
    # run can train a model in one step and load it in another.
    workspace = Workspace(datasets={}, scratch_dir=tmp_path)
    workspace.run_code("open('m.txt', 'w').write('lift=0.1')")
    out = workspace.run_code("result = open('m.txt').read()")
    assert "lift=0.1" in out


def test_run_code_has_no_persistence_without_a_scratch_dir() -> None:
    # The default workspace has no scratch dir, so each call is isolated and a file one
    # writes is gone for the next (the read fails, reported back for the model to read).
    workspace = Workspace(datasets={})
    workspace.run_code("open('m.txt', 'w').write('x')")
    out = workspace.run_code("result = open('m.txt').read()")
    assert "error" in out and "FileNotFoundError" in out


def test_stateful_workspace_keeps_variables_across_run_code_calls() -> None:
    # A stateful workspace runs a persistent session, so a variable from one call is
    # still defined in the next (the notebook-like loop), unlike the one-shot default.
    workspace = Workspace(datasets={"d": _two_groups()}, stateful=True)
    try:
        assert "error:" not in workspace.run_code("total = len(data['d'])")
        assert "result: 1200" in workspace.run_code("result = total")
    finally:
        workspace.close()


def test_non_stateful_workspace_forgets_variables_between_calls() -> None:
    # The default one-shot workspace does not carry in-memory state, so a variable from
    # a prior call is undefined in the next.
    workspace = Workspace(datasets={})
    workspace.run_code("total = 5")
    out = workspace.run_code("result = total")
    assert "error" in out and "NameError" in out


def test_history_and_memory_seed_the_transcript() -> None:
    # A follow-up run replays the prior turns and the durable derivation memory into the
    # transcript, so "it" resolves and the model builds on earlier work: before the new
    # question, and without the oracle ever trusting any of it.
    class _Recorder:
        def __init__(self, *steps: Step) -> None:
            self._steps = list(steps)
            self._i = 0
            self.seen: Transcript | None = None

        def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
            self.seen = transcript
            step = self._steps[self._i]
            self._i += 1
            return step

    client = _Recorder(Step(tool_calls=(_call("answer", summary="tuned it"),)))
    answer(
        "now tune it",
        Workspace(datasets={}),
        client,
        history=[("user", "train a model"), ("assistant", "got R2 0.9 with xgboost")],
        memory="- house_price [sound]: target=price",
    )
    assert client.seen is not None
    blob = "\n".join(str(m["content"]) for m in client.seen.messages)
    assert "train a model" in blob  # prior user turn
    assert "R2 0.9 with xgboost" in blob  # prior assistant turn
    assert "house_price" in blob  # durable derivation memory
    assert "now tune it" in blob  # the new question
    # history comes before the new question
    assert blob.index("train a model") < blob.index("now tune it")


def test_bash_needs_a_stateful_session() -> None:
    # bash runs in the session; a one-shot workspace has none, so it declines.
    out = Workspace(datasets={}).run_bash("echo hi")
    assert "error" in out and "stateful" in out


def test_bash_on_the_host_backend_is_declined() -> None:
    # A stateful host-subprocess session declines bash (a host shell would touch the
    # user's machine); it is a docker-backend capability.
    workspace = Workspace(datasets={}, stateful=True, backend="subprocess")
    try:
        out = workspace.run_bash("echo hi")
        assert "error" in out and "docker" in out
    finally:
        workspace.close()


def test_bash_tool_is_offered_and_dispatched() -> None:
    # The loop exposes a bash tool and routes it to the workspace.
    from elbi_agent.runtime import _TOOLSPECS, _dispatch, _RunState

    assert any(spec.name == "bash" for spec in _TOOLSPECS)
    workspace = Workspace(datasets={})  # not stateful -> declines, but proves routing
    output, terminal = _dispatch(
        _call("bash", command="echo hi"), workspace, 1, _RunState()
    )
    assert terminal is None and "stateful" in output


# -- the model-lifecycle tools (train_model / list_models / predict / promote) -----


def _tool_call(tool: str, arguments: dict[str, Any] | None = None) -> ToolCall:
    # `_call` binds its first parameter as `name`, which collides with tool
    # arguments literally named "name" (train_model's, promote_model's).
    return ToolCall(id=f"c-{tool}", name=tool, arguments=dict(arguments or {}))


def test_train_model_routes_parsed_args_to_the_injected_trainer() -> None:
    # The loop parses and normalizes the call, the injected TrainFn does the work,
    # and its report is what the model reads back.
    calls: list[dict[str, Any]] = []

    def train(spec: Mapping[str, Any]) -> str:
        calls.append(dict(spec))
        return "Trained and registered **churn** version 1"

    workspace = Workspace(datasets={"d": []}, train=train)
    client = _Scripted(
        Step(
            tool_calls=(
                _tool_call(
                    "train_model",
                    {
                        "name": "churn",
                        "dataset": "d",
                        "target": "y",
                        "features": ["x1", " x2 "],
                        "time_budget": 30,
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="Model trained."),)),
    )
    result = answer("train a churn model", workspace, client)
    assert calls == [
        {
            "name": "churn",
            "dataset": "d",
            "source_kind": "dataset",
            "target": "y",
            "features": ("x1", "x2"),
            "task": "auto",
            "time_budget": 30.0,
            "metric": None,
            "ensemble": False,
            "time_col": None,
            "horizon": None,
            "engine": "flaml",
        }
    ]
    assert result.narrative.startswith("Model trained.")


def test_model_tools_decline_without_the_capability() -> None:
    # A workspace without the ml capabilities names the missing extra rather than
    # erroring, exactly like `derive` in a bare workspace.
    from elbi_agent.runtime import _dispatch, _RunState

    workspace = Workspace(datasets={"d": []})
    state = _RunState()
    for call in (
        _tool_call("train_model", {"name": "m", "dataset": "d", "target": "y"}),
        _call("list_models"),
        _call("predict", model="m", rows=[{"x": 1}]),
        _tool_call("promote_model", {"name": "m", "version": 2}),
    ):
        output, terminal = _dispatch(call, workspace, 1, state)
        assert terminal is None and "ml extra" in output


def test_train_model_validates_before_calling_the_trainer() -> None:
    from elbi_agent.runtime import _dispatch, _RunState

    called = False

    def train(*args: object) -> str:
        nonlocal called
        called = True
        return "ok"

    workspace = Workspace(datasets={"d": []}, train=train)
    state = _RunState()
    output, _ = _dispatch(_tool_call("train_model", {"name": "m"}), workspace, 1, state)
    assert "needs `name`, a data source" in output
    output, _ = _dispatch(
        _tool_call(
            "train_model",
            {"name": "m", "dataset": "d", "target": "y", "time_budget": "fast"},
        ),
        workspace,
        1,
        state,
    )
    assert "must be a number" in output
    assert called is False  # invalid calls never reach the trainer


def test_predict_and_promote_validate_their_shapes() -> None:
    from elbi_agent.runtime import _dispatch, _RunState

    predictions: list[tuple[str, tuple[Mapping[str, Any], ...]]] = []
    promotions: list[tuple[str, int, str]] = []
    workspace = Workspace(
        datasets={"d": []},
        predict_model=lambda ref, rows, raw: (
            predictions.append((ref, tuple(rows))) or f"scored {len(rows)}"
        ),
        promote_model=lambda name, version, alias: (
            promotions.append((name, version, alias)) or "promoted"
        ),
    )
    state = _RunState()
    output, _ = _dispatch(
        _call("predict", model="m", rows="not-rows"), workspace, 1, state
    )
    assert "list of feature-record objects" in output and not predictions
    output, _ = _dispatch(
        _call("predict", model="m@champion", rows=[{"x": 1.5}]), workspace, 1, state
    )
    assert output == "scored 1" and predictions == [("m@champion", ({"x": 1.5},))]
    output, _ = _dispatch(
        _tool_call("promote_model", {"name": "m", "version": "two"}),
        workspace,
        1,
        state,
    )
    assert "integer" in output and not promotions
    output, _ = _dispatch(
        _tool_call("promote_model", {"name": "m", "version": 2}), workspace, 1, state
    )
    assert output == "promoted" and promotions == [("m", 2, "champion")]


def test_model_tools_are_offered_to_the_model() -> None:
    from elbi_agent.runtime import _TOOLSPECS

    offered = {spec.name for spec in _TOOLSPECS}
    assert {"train_model", "list_models", "predict", "promote_model"} <= offered
