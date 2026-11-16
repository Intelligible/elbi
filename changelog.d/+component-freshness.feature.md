Served components gained `provenance.inputs_version`, a hash of the input data they
rest on, and `provenance.computed_at`, the UTC time that version was first computed,
so an agent could tell a current statement from an outdated one. `search_components`
returned them as well.
