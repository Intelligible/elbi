# The Open Derivation Spec (ODS)

The Open Derivation Spec (ODS) is the open standard at the heart of "open spec = no
lock-in." It lives in [`spec/`](https://github.com/Intelligible/elbi/tree/main/spec)
and is the authoritative contract the SDK validates against.

- [`derivation.schema.json`](https://github.com/Intelligible/elbi/blob/main/spec/derivation.schema.json):
  the machine-readable contract (JSON Schema 2020-12). **Authoritative.**
- [`derivation.md`](https://github.com/Intelligible/elbi/blob/main/spec/derivation.md):
  normative prose.
- `tests/`: conformance fixtures run by CI; the executable definition of the
  standard.

The SDK loads this schema **as data** and validates against it; it never mirrors
the schema's constraints in Python. The bundled copy in the package is asserted
byte-identical to the canonical `spec/` file by a test, so the two cannot drift.

The spec is versioned on its own SemVer line, independent of the SDK. Each SDK
release declares the spec version it implements via `elbi.SPEC_VERSION`.

```python
import elbi

elbi.SPEC_VERSION  # -> "1.0"

elbi.validate_manifest(my_manifest)  # raises SpecValidationError if invalid
elbi.is_valid_manifest(my_manifest)  # -> bool
```
