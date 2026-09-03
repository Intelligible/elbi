# Oracle verdict rubric

The ground-truth guide reviewers use to label each trap's `expected_verdict`.
It also doubles as the spec for how the oracle itself should behave.

## The three verdicts

- **sound**: the claim survives the gate. The effect is real, identifiable from the
  data at hand, and stable.
- **unsound**: the gate catches a specific, named flaw that invalidates the claim as
  stated.
- **inconclusive**: the data alone cannot decide. Deciding would need information that
  isn't in the rows (a sampling mechanism, feature timing, an unobserved confounder, an
  untestable assumption).

## The golden rule

If settling the verdict requires an assumption or a fact that can't be checked from the
rows, it is **inconclusive**, never a guessed sound/unsound. We only certify what
observational data can identify. A "we can't tell" is a correct, valuable answer, not a
failure.

## Per pitfall

### Confounding (incl. ice-cream / drowning)
- **unsound**: the claimed X to Y effect vanishes or flips sign after conditioning on an
  observed common cause.
- **sound**: the effect keeps its sign and significance after adjusting for the observed
  confounders, and nothing you adjusted for is a collider or mediator.
- **inconclusive**: the suspected confounder isn't in the data, or the only available
  adjustment is non-identifiable (you can't tell from the rows whether conditioning
  removes bias or adds it).

### Simpson's paradox
- **unsound**: the aggregate association reverses or disappears within the confounder's
  subgroups, and the subgroup view is the correct one.
- **sound**: the association holds and does not reverse across the relevant subgroups.
- **inconclusive**: the subgrouping variable that would resolve it isn't in the data.

### Data leakage
- **unsound**: a feature encodes the target or uses information unavailable at prediction
  time, inflating performance (near-perfect feature-to-label signal, or post-outcome
  timing).
- **sound**: performance holds with no feature acting as a proxy for the label through
  forbidden information.
- **inconclusive**: you can't establish a feature's timing or provenance from the rows,
  so you can't tell if it's leakage or a legitimately strong signal.

### Regression to the mean
- **unsound**: the "effect" is on a group picked for an extreme baseline and it just
  reverts toward the mean, with no real intervention effect.
- **sound**: the change is larger than RTM alone predicts (a control group or a
  baseline-adjusted comparison survives).
- **inconclusive**: you can't tell whether the group was selected on the baseline. That
  selection fact isn't in the rows.

### Multiverse / fragility
- **unsound (fragile)**: the sign or significance flips across a meaningful share of
  defensible analytic specifications, so a single point claim isn't supported.
- **sound (robust)**: the effect holds its sign across the specification curve (and
  mostly its significance).
- **inconclusive**: too few valid specs to judge, or the specs disagree because of a real
  identification ambiguity rather than analyst choices. Report the range, don't force a
  verdict.

### Selection / Berkson
- **unsound**: conditioning on a selection variable that is a common effect of X and Y
  induces a spurious association in the selected sample, and the declared sampling
  mechanism confirms it.
- **sound**: the association is not a selection artifact once the sampling mechanism is
  accounted for.
- **inconclusive (the default here)**: the sampling or selection mechanism is undeclared
  or unknowable from the rows. Do not guess.

## How reviewers apply it

- Two reviewers label independently, then reconcile. If they can't agree, the verdict is
  **inconclusive** by default and the trap gets a note explaining the disagreement.
- Prefer **inconclusive** over a confident wrong label. A trap mislabeled sound/unsound
  poisons the whole suite.
- Every trap records which rule fired and why (the manifest `rationale`), so the label is
  auditable later.
