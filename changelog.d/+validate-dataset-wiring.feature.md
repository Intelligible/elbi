`elbi validate` now checks that `elbi.yaml` and `elbi.dev.yaml` agree when
`elbi.dev.yaml` is present. A declared dataset with no binding, a binding for an
undeclared dataset, and a file binding that does not point at an existing file each fail
validation; before, they surfaced only when a derivation read the dataset, or not at
all. A project that binds a dataset without listing it under `datasets:` now fails
locally where it got at most a warning, and listing the dataset fixes it. Projects
without `elbi.dev.yaml`, such as in CI, skip the check.
