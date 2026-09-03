# Result history

Every time a derivation is certified, it produces a number: the verification oracle's
estimate for the claim, a held-out skill, an effect size. A derivation keeps those
results as a **history**, a changelog of its certified answer, so you can see how the
answer changed as the data, the code, or the controls changed, and see *what moved it*.

This is a property of a derivation, not a separate object. A derivation is your certified
answer to a question; its history tells you whether that answer is stable, and when and
why it moved. Re-asking a question gives you a fresh answer, but only the history
remembers the sequence: that the effect was `+0.45` last quarter and is `+0.39` now, and
that it dropped when the data refreshed rather than when you changed the model.

The difference from a conventional tracker is what the numbers are. MLflow or Weights &
Biases record whatever metric your training code reports; they trust the number you hand
them. Here every number in the history is the oracle's *certified* estimate:
leakage-free, reproducible, and carried with the gates that passed. So the history is a
record of trustworthy results, which matters most when an agent, not a person, authored
the analysis: an agent can log an inflated number as easily as a human, and the oracle is
what keeps the history honest.

## How a version is identified

A derivation is deterministic and content-addressed: its **version** is a hash of its
code, its params, and its input versions. Re-certifying with everything unchanged yields
the same version and collapses to the same entry, so a deterministic re-run adds no noise.
A change to any input yields a new version, so a derivation's history is exactly its
sequence of distinct certified versions over time.

Because the version decomposes into its components, the history can attribute a moved
estimate to its cause. Comparing a version to the one before it reports which dimensions
changed:

- **data**: an input dataset version changed (a refresh, more rows)
- **code**: the derivation's source changed (a different transform or model)
- **controls**: the adjustment set changed (a covariate added or dropped)
- **params**, **claim**: a parameter value or the declared column roles changed

So a data scientist sees not just *"the effect dropped from +0.45 to +0.39"* but *"...and
it dropped because the data refreshed, not because I changed the analysis."*

## Using it

The app shows a derivation's result history on its detail page: each certified version
with its verdict, estimate, date, and, for each, what changed since the previous one.
Over HTTP:

```
GET /api/derivations/{name}/history
```

The CLI shows the same for a local project:

```bash
elbi history <name>
```

## Exporting to MLflow

If your team already runs MLflow, each certified version can be exported to it, so the
certification shows up in the tool you use. This is a thin *client* on MLflow's public
API, not a second tracker: each derivation maps to one MLflow experiment (its versions
group there), the attestation rides as run tags (`elbi.verdict`,
`elbi.data_hash`, `elbi.checks`), and the certified estimate becomes the
run metric, all queryable in MLflow's UI (`tags."elbi.verdict" = 'sound'`).

The client ships with elbi (the skinny, client-only MLflow), so nothing extra to
install, just point it at your server:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000   # or set it in the app settings
```

The export is opt-in (it runs only when a URI is set) and **fail-soft**: an unreachable
server never affects certification, and the version is recorded in the local history
either way. Only certified versions are exported, with the verdict encoded explicitly, so
a tag can never be misread as sound.

## What this is not

This is not a general experiment tracker, and does not try to beat MLflow or W&B at
hyperparameter sweeps, artifact storage, a model-stage registry, or automated retraining
pipelines; those are commodity MLOps and well served elsewhere. The one thing added, and
the only thing worth adding, is that the numbers are certified by construction and the
history says what moved them. The unit of record is the content-addressed derivation
version itself, sealed at certification; there is no mutable run to edit, no artifact blob
store, and no self-reported metric to distrust.
