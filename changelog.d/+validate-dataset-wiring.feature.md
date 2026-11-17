`elbi validate` now checks that `elbi.yaml` and `elbi.dev.yaml` agree when
`elbi.dev.yaml` is present. A declared dataset with no binding, a binding for a dataset
that is not declared, and a file binding whose file is missing each fail validation,
where before they surfaced only when a derivation happened to read the dataset, or not
at all. Projects without `elbi.dev.yaml`, such as in CI, skip the check.
