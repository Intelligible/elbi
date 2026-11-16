Notebooks gained a `run("<derivation>", **params)` helper next to `data`. It returns
the value the project's own derivation computes, so a notebook can reuse governed
logic instead of copying its maths into cells.
