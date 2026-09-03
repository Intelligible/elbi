# Verification certificates

A certified derivation's answer is only as trustworthy as the system that produced it -
until someone downstream can check it themselves. A **certificate** is a signed,
exportable statement of what the verification oracle certified for one derivation: the
verdict, the gates that ran (and any that were skipped, and why), the certified estimate,
the content-addressed identity of the code and data, and who certified it, when. Anyone
holding the certificate and the certifier's public key can verify it offline, with no
running system and no access to your data - the point of a certificate is that trust
travels with the file.

## How it is issued and verified

Tamper-evidence is a detached Ed25519 signature over the certificate's canonical JSON
bytes. The signature travels in an envelope beside the payload:

```json
{
  "schema": "elbi.certificate/v1",
  "certificate": { "...the signed payload...": "..." },
  "signature": { "algorithm": "ed25519", "value": "<base64>" }
}
```

The signing key is a 32-byte seed persisted at `.elbi/certificate.key` (created on
first use, mode `0o600`), with the base64 public key written beside it as
`certificate.key.pub` for distribution - or point `ELBI_CERTIFICATE_KEY` at a path
of your choosing for an app or CI deployment. Because the signing public key rides inside
the signed payload too, an attacker who edits a certificate and re-signs it with their own
key produces a document that is self-consistent on its own terms - this is why
verification against a **separately distributed** public key matters: only a verifier who
already knows the real key can catch a re-signed forgery. A verification run with no
public key pinned only proves the document is internally self-consistent, not that it
came from a key you trust.

A certificate is only ever issued for a derivation with an actual verified claim: one the
oracle checked and returned a real verdict for. A derivation that merely ran without
error, with no claim ever declared, has nothing to certify and is not issued one.

## Using it

The app shows JSON and PDF download links on a certified derivation's detail page, and
inline in the chat view beside a verified finding. Over HTTP:

```
GET  /api/certificates/{name}        the signed certificate, as JSON
GET  /api/certificates/{name}/pdf    the certificate rendered as a one-page PDF
POST /api/certificates/{name}        verify a certificate against this derivation
                                      (400 if the signature is invalid, 409 if it is
valid but stale), the sync gate a pull/push
                                      round trip uses to reject a tampered or
                                      out-of-date file
```

The CLI verifies (and issues) certificates fully offline, with no project or running system
needed to check one someone sent you:

```bash
elbi certificate issue <name>                     # sign the newest certified run
elbi certificate verify <file> --public-key <key> # verify against a trusted key
elbi certificate chain                            # verify the hash-chained audit log
elbi certificate public-key                       # print this project's public key
```

## The audit log

Alongside certificates, a project can keep a hash-chained, append-only log of every
served invocation: each line carries the SHA-256 of the line before it, so editing,
reordering, or dropping a record breaks the chain. `elbi certificate chain` (or
its HTTP equivalent) verifies it offline and reports a head hash you can anchor
externally - the one gap a chain cannot see on its own is truncation of its own newest
records, which is exactly what an external anchor catches. See
[Serving over MCP](./mcp.md#the-audit-trail) for how the audit sink is
wired into a running server.

## What this is not

A certificate is not part of the
[Open Derivation Spec](https://github.com/Intelligible/elbi/blob/main/spec/derivation.md):
the spec explicitly leaves signatures and certification governance out of scope, as a
deployment concern, so there is no plan to standardize the certificate format there. The PDF is a
human-readable companion for an auditor, not a second source of truth - always verify the
JSON certificate (or its embedded signature) for anything that matters; the PDF exists so
a person can glance at a finding without running a tool. And the audit log is a separate
integrity mechanism: a certificate proves one derivation's result was checked and signed;
the audit log proves the *sequence of invocations* that produced it was not tampered with
after the fact. Neither substitutes for the other.
