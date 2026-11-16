Added `verify_pooled_effect`, which checks an effect pooled across several past
studies (one effect and standard error each) before it is trusted. It reports the
fixed- and random-effects estimates, the between-study spread, and the Hartung-Knapp
and prediction intervals, and it flags a pool of fewer than five studies, a result
one study carries, and a new study whose result conflicts with the pooled estimate
used as its prior. A derivation that declares `{"effect": ..., "se": ...}` as its
claim can now be certified.
