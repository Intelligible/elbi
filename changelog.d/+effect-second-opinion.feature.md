`elbi_core.verification.second_opinion` gives an independent estimate of an average
effect (ATE, ATT or ATC) from the rows: g-computation under three outcome models with
bootstrap standard errors, plus a cross-fitted doubly robust estimate for a two-valued
treatment. `combine_interval` and `combine_point` combine it with an agent's own interval
or point estimate, equally weighted, so a disagreement between the two analyses shows up
in the answer instead of being resolved by whichever ran last. Requires the `ml` extra.
