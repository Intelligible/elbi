"""Scoped credentials, and the two CVE classes they exist to avoid.

Apache Polaris, the reference implementation of this pattern, shipped both of these as
critical CVEs in one release, so they are tested as properties rather than as examples:
the interesting inputs are exactly the ones nobody writes by hand.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elbi_core.sandbox import (
    MAX_TTL,
    AwsCredentialBroker,
    AzureCredentialBroker,
    CredentialError,
    GcsCredentialBroker,
    StorageScope,
    blob_sas_terms,
    broker_for,
    cel_literal,
    gcs_access_boundary,
    s3_session_policy,
    scope_for,
    storage_prefix,
)

#: Names a person might reasonably give a dataset, plus the ones an attacker would.
_NAMES = st.text(
    alphabet=st.characters(
        min_codepoint=32, max_codepoint=126, blacklist_characters="\x00"
    ),
    min_size=1,
    max_size=40,
)


def test_a_prefix_always_ends_at_a_boundary() -> None:
    # The whole point: a grant on `data` must not also reach `data_secrets`.
    assert storage_prefix("data") == "data/"
    assert storage_prefix("data/") == "data/"
    assert storage_prefix("  warehouse/orders  ") == "warehouse/orders/"


@given(name=_NAMES)
def test_a_prefix_is_either_bounded_or_refused(name: str) -> None:
    """Every accepted prefix ends in a separator, whatever the name contained.

    This is the property that makes ``arn:...:bucket/<prefix>*`` a bound. A prefix that
    did not end at a separator would make the trailing wildcard match a sibling.
    """
    try:
        result = storage_prefix(name)
    except CredentialError:
        return  # refusing is always an acceptable answer
    assert result.endswith("/")
    assert "*" not in result and "?" not in result


def test_a_wildcard_in_a_name_is_refused_rather_than_interpolated() -> None:
    # CVE-2026-42810: a literal `*` in a table name reused in an IAM resource pattern
    # gave access across tables. IAM has no escape for it, so the only sound answer is
    # to refuse.
    for hostile in ("orders*", "orders?", "*", "a*b"):
        with pytest.raises(CredentialError, match="wildcard"):
            storage_prefix(hostile)


def test_a_prefix_cannot_climb_out_of_its_scope() -> None:
    with pytest.raises(CredentialError, match=r"'\.\.'"):
        storage_prefix("warehouse/../secrets")


def test_a_control_character_never_reaches_a_policy() -> None:
    for hostile in ("or\nders", "orders\x00", "or\tders"):
        with pytest.raises(CredentialError):
            storage_prefix(hostile)
        with pytest.raises(CredentialError):
            cel_literal(hostile)
    # Surrounding whitespace is a typo rather than an attack, and stripping it leaves a
    # name that is safe by the same check.
    assert storage_prefix("orders\n") == "orders/"


@given(name=_NAMES)
def test_a_cel_literal_cannot_be_escaped_from(name: str) -> None:
    """A CEL literal stays one literal, whatever the name contains.

    CVE-2026-42811 was a single quote in a table name closing the literal and collapsing
    a prefix restriction to bucket-wide read and write. The property that prevents it:
    after the opening and closing quotes are removed, no unescaped quote remains.
    """
    try:
        literal = cel_literal(name)
    except CredentialError:
        return
    assert literal.startswith("'") and literal.endswith("'")
    body = literal[1:-1]
    # Walk the body, consuming escapes. A quote that survives is a quote that would have
    # ended the literal early.
    index = 0
    while index < len(body):
        if body[index] == "\\":
            index += 2
            continue
        assert body[index] != "'", f"{name!r} escaped its literal: {literal!r}"
        index += 1


def test_the_cel_escape_is_ordered_so_a_backslash_is_not_re_escaped() -> None:
    # Escaping the quote first would turn `a\` + `'` into `a\\'`, in which the backslash
    # protects the backslash and the quote is live again.
    assert cel_literal("a\\'b") == "'a\\\\\\'b'"
    assert cel_literal("plain") == "'plain'"


def test_the_s3_policy_grants_objects_and_listing_separately() -> None:
    scope = StorageScope(bucket="warehouse", prefixes=("orders",))
    policy = s3_session_policy(scope)
    objects, listing = policy["Statement"]
    assert objects["Resource"] == ["arn:aws:s3:::warehouse/orders/*"]
    assert objects["Action"] == ["s3:GetObject"]
    # Listing is a bucket-level action, so it is granted on the bucket and bounded by a
    # condition. Granting only the object statement would block listing entirely;
    # granting ListBucket without the condition would expose the whole bucket to it.
    assert listing["Resource"] == ["arn:aws:s3:::warehouse"]
    assert listing["Condition"]["StringLike"]["s3:prefix"] == ["orders/*"]


def test_write_access_is_opt_in() -> None:
    read = s3_session_policy(StorageScope(bucket="w", prefixes=("orders",)))
    assert read["Statement"][0]["Action"] == ["s3:GetObject"]
    write = s3_session_policy(
        StorageScope(bucket="w", prefixes=("orders",), read_only=False)
    )
    assert "s3:PutObject" in write["Statement"][0]["Action"]


def test_a_policy_too_large_for_aws_fails_before_the_api_call() -> None:
    # AWS budgets 2,048 characters across the inline policy and any managed policy ARNs,
    # and the packed-binary limit that actually fires gives a far less useful message.
    scope = StorageScope(
        bucket="warehouse", prefixes=tuple(f"table-{n:04d}" for n in range(200))
    )
    broker = AwsCredentialBroker(role_arn="arn:aws:iam::1:role/sandbox")
    with pytest.raises(CredentialError, match="over the 2048"):
        broker.vend(scope)


def test_a_long_lived_credential_is_refused() -> None:
    broker = AwsCredentialBroker(role_arn="arn:aws:iam::1:role/sandbox")
    scope = StorageScope(bucket="w", prefixes=("orders",))
    with pytest.raises(CredentialError, match="exceeds"):
        broker.vend(scope, ttl=MAX_TTL + dt.timedelta(minutes=1))
    with pytest.raises(CredentialError, match="positive"):
        broker.vend(scope, ttl=dt.timedelta(0))


def test_the_gcs_boundary_carries_both_condition_forms() -> None:
    scope = StorageScope(bucket="warehouse", prefixes=("orders",))
    boundary = gcs_access_boundary(scope)
    (rule,) = boundary["accessBoundary"]["accessBoundaryRules"]
    assert rule["availableResource"] == (
        "//storage.googleapis.com/projects/_/buckets/warehouse"
    )
    assert rule["availablePermissions"] == ["inRole:roles/storage.objectViewer"]
    assert "startsWith(" in rule["availabilityCondition"]["expression"]
    # A resource.name condition cannot gate a list: for a list the resource named is the
    # bucket. Without the list prefix the boundary either blocks listing or exposes the
    # whole bucket to it.
    assert boundary["objectListPrefixes"] == ["orders/"]


def test_the_gcs_boundary_refuses_more_rules_than_google_accepts() -> None:
    scope = StorageScope(bucket="w", prefixes=tuple(f"t{n}" for n in range(11)))
    with pytest.raises(CredentialError, match="at most 10 rules"):
        gcs_access_boundary(scope)


def test_a_hostile_name_produces_a_boundary_that_still_scopes() -> None:
    # The end-to-end version of the CEL property: the quote survives as data.
    scope = StorageScope(bucket="w", prefixes=("orders'; //",))
    boundary = gcs_access_boundary(scope)
    expression = boundary["accessBoundary"]["accessBoundaryRules"][0][
        "availabilityCondition"
    ]["expression"]
    assert expression.count("startsWith(") == 1
    assert expression.endswith("')")
    # One expression, one literal: the injected quote did not close it.
    assert expression.count("'") - expression.count("\\'") == 2


def test_the_sas_terms_scope_to_a_directory() -> None:
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    terms = blob_sas_terms(
        StorageScope(bucket="warehouse", prefixes=("orders",)),
        ttl=dt.timedelta(minutes=15),
        now=now,
    )
    # sr=d is what makes the prefix the boundary rather than a single blob.
    assert terms["resource"] == "d"
    assert terms["blob_name"] == "orders"
    assert terms["permission"] == "r"
    assert terms["expiry"] == now + dt.timedelta(minutes=15)
    with pytest.raises(CredentialError, match="one directory"):
        blob_sas_terms(
            StorageScope(bucket="w", prefixes=("a", "b")), ttl=dt.timedelta(minutes=5)
        )


def test_gcs_downscoping_reaches_the_sdk_with_the_boundary_it_was_given() -> None:
    """The boundary is what bounds the credential, so it is what has to arrive intact.

    Mocked at the SDK edge rather than the network: what matters is that the rules built
    here, including the escaped CEL expression, are the rules handed to google-auth. A
    test against a live project would prove less, because a wrong boundary still returns
    a working token.
    """
    from google.auth import downscoped

    source = SimpleNamespace()
    captured: dict[str, Any] = {}

    class _Credentials:
        def __init__(self, *, source_credentials: Any, credential_access_boundary: Any):
            captured["source"] = source_credentials
            captured["boundary"] = credential_access_boundary
            self.token = "downscoped-token"
            self.expiry = dt.datetime(2026, 1, 1, 12, 0)

        def refresh(self, request: Any) -> None:
            captured["refreshed"] = True

    with mock.patch.object(downscoped, "Credentials", _Credentials):
        vended = GcsCredentialBroker(source_credentials=source).vend(
            StorageScope(bucket="warehouse", prefixes=("orders'; //",))
        )

    assert captured["source"] is source
    assert captured["refreshed"] is True
    (rule,) = captured["boundary"].rules
    expression = rule.availability_condition.expression
    # The injected quote survived as data: one startsWith, one literal.
    assert expression.count("startsWith(") == 1
    assert expression.count("'") - expression.count("\\'") == 2
    assert vended.env["GOOGLE_OAUTH_ACCESS_TOKEN"] == "downscoped-token"
    # The listing bound travels alongside, because a resource.name condition cannot gate
    # a
    # list and losing it would silently widen what a session can enumerate.
    # Already ends in a separator, so the normaliser leaves it: the invariant is that it
    # ends at one, not that another is appended.
    assert vended.env["ELBI_STORAGE_PREFIXES"] == "orders'; //"
    # google-auth reports a naive expiry; everything downstream compares against an
    # aware
    # one, so a naive value must not escape.
    assert vended.expires_at.tzinfo is not None


def test_gcs_inherits_its_expiry_rather_than_inventing_one() -> None:
    # A downscoped token has no lifetime of its own. Claiming the requested ttl would be
    # a
    # lie about when the credential stops working.
    from google.auth import downscoped

    class _Credentials:
        def __init__(self, **_: Any) -> None:
            self.token = "t"
            self.expiry = None

        def refresh(self, request: Any) -> None:
            pass

    with mock.patch.object(downscoped, "Credentials", _Credentials):
        vended = GcsCredentialBroker(source_credentials=SimpleNamespace()).vend(
            StorageScope(bucket="w", prefixes=("orders",)), ttl=dt.timedelta(minutes=10)
        )
    # No expiry from the source: falls back to the requested window rather than to
    # "never".
    assert 0 < vended.seconds_remaining() <= 600


def test_azure_signs_with_a_delegation_key_and_never_the_account_key() -> None:
    """An account-key SAS cannot be revoked without rotating the key everyone shares.

    So the test is about *which* key signs it, and that the directory scope (``sdd``) is
    the depth of the prefix; that is what makes the prefix the boundary rather than one
    blob.
    """
    from azure.storage import blob as azure_blob

    captured: dict[str, Any] = {}

    class _Service:
        def __init__(self, url: str, credential: Any) -> None:
            captured["url"] = url
            captured["credential"] = credential

        def get_user_delegation_key(self, key_start_time: Any, key_expiry_time: Any):
            captured["key_window"] = (key_start_time, key_expiry_time)
            return "delegation-key"

    def _generate(**kwargs: Any) -> str:
        captured["sas"] = kwargs
        return "sv=2025-01-01&sig=abc"

    with (
        mock.patch.object(azure_blob, "BlobServiceClient", _Service),
        mock.patch.object(azure_blob, "generate_blob_sas", _generate),
    ):
        vended = AzureCredentialBroker("acct", credential="entra").vend(
            StorageScope(bucket="warehouse", prefixes=("lake/orders",)),
            ttl=dt.timedelta(minutes=15),
        )

    sas = captured["sas"]
    assert sas["user_delegation_key"] == "delegation-key"
    assert "account_key" not in sas
    assert sas["permission"] == "r"
    assert sas["blob_name"] == "lake/orders"
    # sdd is the directory depth, which is what scopes the SAS to a prefix.
    assert sas["sdd"] == 2
    # The delegation key's window brackets the SAS, so clock skew cannot expire the key
    # before the token it signed.
    start, expiry = captured["key_window"]
    assert start < sas["start"] and expiry > sas["expiry"]
    assert vended.env["AZURE_STORAGE_SAS_TOKEN"].startswith("sv=")
    assert vended.expires_at == sas["expiry"]


def test_the_broker_is_chosen_by_the_storage_scheme() -> None:
    assert isinstance(
        broker_for("s3://warehouse", role_arn="arn:aws:iam::1:role/s"),
        AwsCredentialBroker,
    )
    assert isinstance(broker_for("gs://warehouse"), GcsCredentialBroker)
    with pytest.raises(CredentialError, match="no credential broker"):
        broker_for("file:///tmp/warehouse")


def test_a_scope_is_rooted_where_the_warehouse_is() -> None:
    scope = scope_for("s3://warehouse/lakehouse", prefixes=["orders", "customers"])
    assert scope.bucket == "warehouse"
    assert scope.prefixes == ("lakehouse/orders/", "lakehouse/customers/")
    policy = json.dumps(s3_session_policy(scope))
    assert "arn:aws:s3:::warehouse/lakehouse/orders/*" in policy


def test_session_policy_stays_within_a_pod_identity_budget() -> None:
    """The policy has to fit beside the caller's session tags, not just under 2,048.

    AWS compresses the session policy and the caller's session tags into one budget. A
    pod reaching this code through EKS Pod Identity brings six transitive tags, and a
    real deployment measured roughly three hundred characters left for the policy, so
    the documented 2,048 is not the number to design against. This pins the size of the
    common case, which is one prefix under a warehouse root in a bucket whose name
    carries an account id.
    """
    scope = StorageScope(bucket="elbi-warehouse-123456789012", prefixes=("warehouse/",))
    encoded = json.dumps(s3_session_policy(scope), separators=(",", ":"))
    assert len(encoded) <= 330, (
        f"the scoped policy grew to {len(encoded)} characters. Measured against a real "
        "Pod Identity session, 375 was refused and 326 was accepted, so growth here "
        "breaks credential vending on EKS rather than merely being untidy."
    )


def test_session_policy_carries_no_sid_fields() -> None:
    """Sids are documentation, and this document is measured rather than read."""
    policy = s3_session_policy(StorageScope(bucket="w", prefixes=("orders",)))
    assert all("Sid" not in statement for statement in policy["Statement"])
