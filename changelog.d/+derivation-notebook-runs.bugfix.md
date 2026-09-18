Opening a derivation in a notebook now produces something that runs. Three faults
stacked: a derivation's source was read back with `inspect.getsource(fn)`, which returns
the function and discards the module its imports, constants, contract and helpers live
in; a relative import between derivation modules was copied verbatim into a cell, where
there is no parent package to resolve it against; and `@derivation` raised
`DuplicateDerivationError` on a second run, so a derivation cell could be executed
exactly once — including by the reactive engine, which re-runs cells unprompted. The
source is now captured with the module-level context it depends on, relative imports are
followed into the sibling module and emitted as definitions, and a notebook kernel
registers into a registry where re-declaring replaces. Discovery still fails loudly on a
duplicate name, which is a project error rather than an edit.

The notebook also shows what the derivation returned. Defining a derivation displays
nothing — a decorated `def` is a statement — so the seeded notebook now ends on a cell
that resolves the derivation's inputs the way the runner does (a dataset from the
notebook's own data, an upstream derivation by running it first) and calls it. An
`Artifact` renders itself: a table draws as a table and markdown as markdown, rather
than as a dataclass repr. Notebook markdown no longer reads `$` as inline TeX, which
was turning every dollar amount into an integrand.
