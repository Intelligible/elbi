"""Short-lived, prefix-scoped credentials for a sandbox to read object storage itself.

A kernel that fetches data on demand needs to reach the warehouse. Two ways exist: proxy
every byte through the app, or hand the sandbox a credential that can read exactly what
that session is entitled to. The second is chosen: the app stops being a data-plane
bottleneck, large scans stay off its network path, and authorisation is audited once
where the credential is minted rather than per byte.

That trade only holds if the scoping is airtight, and this is a place where the
reference implementation for this exact pattern got it wrong twice in one release.
Apache Polaris 1.4.0 shipped two critical CVEs on 2026-05-04:

* `CVE-2026-42811 <https://github.com/advisories/GHSA-fc3h-c6h7-r83j>`_: a table path
  interpolated unescaped into a GCS Credential Access Boundary CEL expression. A name
  containing a single quote closed the literal and collapsed the restriction to
  **bucket-wide read and write**.
* `CVE-2026-42810 <https://github.com/advisories/GHSA-vxgg-mqx2-3w59>`_: a literal
  ``*`` in a table name reused in S3 IAM resource patterns and ``s3:prefix``, giving
  access across tables.

Both are the same bug: user-controlled text placed into a language that gives some
characters meaning. Dataset and notebook names here are user-controlled and reach
exactly these boundaries, so the two defences are structural rather than advisory:
:func:`cel_literal` for anything entering a CEL expression, and :func:`storage_prefix`
for anything becoming a path, which also guarantees the trailing separator that stops a
grant on ``data/`` from authorising ``data_foo/``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import ElbiError


class CredentialError(ElbiError):
    """Scoped credentials could not be minted, or were asked for unsafely."""


#: The longest a vended credential may live. Short because the blast radius of a leaked
#: one is "everything that prefix holds, until it expires", and because a session that
#: outlives it renews rather than holding one credential for a working day.
MAX_TTL = dt.timedelta(hours=1)
DEFAULT_TTL = dt.timedelta(minutes=15)

#: Characters IAM reads as wildcards in a resource ARN. IAM has no escape for them, so a
#: prefix containing one cannot be expressed as a bounded grant at all.
_ARN_WILDCARDS = ("*", "?")

#: A control character in a prefix is either an attempt to break out of a policy
#: document or a corrupted name; neither should reach a cloud API.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def storage_prefix(prefix: str) -> str:
    """Normalise a storage prefix, or refuse to grant on it.

    Guarantees the property that makes a prefix grant mean what it looks like: it ends
    in a separator, so a grant on ``data/`` cannot also authorise ``data_foo/``. Refuses
    wildcards, because IAM resource patterns give them meaning and provide no way to
    escape them, the failure mode behind CVE-2026-42810, and refuses ``..``, which would
    otherwise let a name climb out of the prefix it was scoped to.
    """
    cleaned = prefix.strip()
    if not cleaned:
        raise CredentialError("a credential must be scoped to a non-empty prefix")
    if _CONTROL.search(cleaned):
        raise CredentialError("a storage prefix must not contain control characters")
    for wildcard in _ARN_WILDCARDS:
        if wildcard in cleaned:
            raise CredentialError(
                f"storage prefix {prefix!r} contains {wildcard!r}, which IAM reads "
                "as a wildcard and offers no way to escape. Refusing to mint a "
                "credential whose scope would be wider than it appears."
            )
    if any(part == ".." for part in cleaned.split("/")):
        raise CredentialError(f"storage prefix {prefix!r} must not contain '..'")
    return cleaned if cleaned.endswith("/") else cleaned + "/"


def cel_literal(value: str) -> str:
    """``value`` as a single-quoted CEL string literal, escaped.

    The direct fix for CVE-2026-42811. Backslash first, then the quote, so an escape
    introduced by the first substitution is not re-escaped by the second. Control
    characters are refused rather than encoded: a newline inside a boundary expression
    has no legitimate meaning here, and refusing is a smaller surface than encoding.
    """
    if _CONTROL.search(value):
        raise CredentialError("a CEL literal must not contain control characters")
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


@dataclass(frozen=True, slots=True)
class StorageScope:
    """What a session is allowed to reach: prefixes within one bucket or container."""

    bucket: str
    prefixes: tuple[str, ...]
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise CredentialError("a storage scope needs a bucket")
        if not self.prefixes:
            raise CredentialError("a storage scope needs at least one prefix")
        object.__setattr__(
            self, "prefixes", tuple(storage_prefix(p) for p in self.prefixes)
        )


@dataclass(frozen=True, slots=True)
class VendedCredentials:
    """A minted credential, as the environment a sandbox receives it in."""

    env: Mapping[str, str]
    expires_at: dt.datetime
    scope: StorageScope

    def seconds_remaining(self, now: dt.datetime | None = None) -> float:
        """How long this credential is still good for."""
        moment = now or dt.datetime.now(dt.timezone.utc)
        return max(0.0, (self.expires_at - moment).total_seconds())


class CredentialBroker(Protocol):
    """Mints a credential scoped to one session's storage scope."""

    def vend(
        self, scope: StorageScope, *, ttl: dt.timedelta = DEFAULT_TTL
    ) -> VendedCredentials:
        """A credential good for ``scope`` and no wider, expiring within ``ttl``."""
        ...


def _checked_ttl(ttl: dt.timedelta) -> int:
    """``ttl`` in seconds, bounded by :data:`MAX_TTL`."""
    if ttl <= dt.timedelta(0):
        raise CredentialError("a credential ttl must be positive")
    if ttl > MAX_TTL:
        raise CredentialError(
            f"a credential ttl of {ttl} exceeds the {MAX_TTL} maximum; a session that "
            "runs longer renews rather than holding one credential for the day"
        )
    return int(ttl.total_seconds())


# -- AWS ---------------------------------------------------------------------------


def s3_session_policy(scope: StorageScope) -> dict[str, Any]:
    """An STS session policy granting exactly ``scope``.

    A session policy can only narrow what the role already allows, so this is the second
    of two bounds rather than the only one. Both statements are needed and neither is
    sufficient: object access is granted on ``arn:...:bucket/prefix*``, while listing is
    a *bucket*-level action gated by an ``s3:prefix`` condition, so a policy with only
    the first blocks listing, and one with only the second exposes the whole bucket to
    it.

    The prefixes are wildcard-free by construction (:func:`storage_prefix` refuses the
    characters IAM would read as wildcards), which is what keeps ``prefix*`` a bound
    rather than an invitation.

    Every byte here is spent against a budget the caller does not control, so the policy
    carries no ``Sid`` fields: they are documentation for a policy nobody reads, written
    into a request that is measured. See :data:`_SESSION_POLICY_LIMIT`.
    """
    objects = [f"arn:aws:s3:::{scope.bucket}/{p}*" for p in scope.prefixes]
    actions = ["s3:GetObject"]
    if not scope.read_only:
        actions += ["s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": actions,
                "Resource": objects,
            },
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{scope.bucket}"],
                "Condition": {
                    "StringLike": {"s3:prefix": [f"{p}*" for p in scope.prefixes]}
                },
            },
        ],
    }


#: AWS caps an inline session policy at 2,048 characters, and that is not the limit that
#: bites. It also compresses the policy *and the caller's session tags* into one packed
#: budget, so the space actually available depends on who is asking rather than on
#: anything this code can see.  The caller that matters is a pod. EKS Pod Identity
#: stamps six transitive tags on the app's session (cluster name, cluster ARN,
#: namespace, service account, pod name, pod UID) and they are carried into every role
#: assumed from it. Measured against a real deployment they leave room for roughly three
#: hundred characters, against a documented limit of two thousand. A scoped policy for
#: one prefix is about that size, which is why this one is written tersely and why the
#: check below cannot be the only guard.
_SESSION_POLICY_LIMIT = 2048


class AwsCredentialBroker:
    """Vends S3 credentials with ``AssumeRole`` plus an inline session policy."""

    def __init__(
        self,
        role_arn: str,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
        session_name: str = "elbi-sandbox",
    ) -> None:
        self._role_arn = role_arn
        self._region = region
        self._endpoint_url = endpoint_url
        self._session_name = session_name

    def vend(
        self, scope: StorageScope, *, ttl: dt.timedelta = DEFAULT_TTL
    ) -> VendedCredentials:
        """Assume the role down to ``scope`` and return the session's credentials."""
        policy = json.dumps(s3_session_policy(scope), separators=(",", ":"))
        if len(policy) > _SESSION_POLICY_LIMIT:
            raise CredentialError(
                f"the session policy for this scope is {len(policy)} characters, over "
                f"the {_SESSION_POLICY_LIMIT} AWS allows. Scope the session to fewer "
                "prefixes, or grant on a common parent."
            )
        try:
            import boto3
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised by the extra's absence
            raise CredentialError(
                "vending AWS credentials needs boto3; install elbi[aws]"
            ) from exc

        client = boto3.client(
            "sts", region_name=self._region, endpoint_url=self._endpoint_url
        )
        try:
            response = client.assume_role(
                RoleArn=self._role_arn,
                RoleSessionName=self._session_name,
                Policy=policy,
                DurationSeconds=_checked_ttl(ttl),
            )
        except client.exceptions.PackedPolicyTooLargeException as exc:
            # The character check above passed and AWS refused anyway, which means the
            # caller's own session tags took the space. Saying so is the whole value of
            # catching this: the raw error reports a percentage of a budget it does not
            # name, against a policy the operator can see is well inside the documented
            # limit, and reads as a bug in this code rather than a property of the
            # session it was called from.
            raise CredentialError(
                f"AWS refused the session policy for this scope, though at "
                f"{len(policy)} characters it is inside the {_SESSION_POLICY_LIMIT} it "
                "documents. A session policy shares one compressed budget with the "
                "session tags of whoever is asking, and on EKS Pod Identity those tags "
                "take most of it. Grant on a common parent prefix instead of several, "
                "or give the app an IRSA role, whose sessions carry no tags."
            ) from exc
        credentials = response["Credentials"]
        return VendedCredentials(
            env={
                "AWS_ACCESS_KEY_ID": credentials["AccessKeyId"],
                "AWS_SECRET_ACCESS_KEY": credentials["SecretAccessKey"],
                "AWS_SESSION_TOKEN": credentials["SessionToken"],
                **({"AWS_REGION": self._region} if self._region else {}),
            },
            expires_at=credentials["Expiration"],
            scope=scope,
        )


# -- GCP ---------------------------------------------------------------------------

#: Cloud Storage roles, named as roles because a Credential Access Boundary takes roles
#: rather than individual permissions.
_GCS_READ_ROLE = "inRole:roles/storage.objectViewer"
_GCS_WRITE_ROLE = "inRole:roles/storage.objectAdmin"

#: A boundary holds at most ten rules, and each prefix needs one rule, so this is the
#: real ceiling on how many prefixes one vended credential can cover.
_MAX_BOUNDARY_RULES = 10


def gcs_access_boundary(scope: StorageScope) -> dict[str, Any]:
    """A Credential Access Boundary granting exactly ``scope``.

    Two condition forms, both required. ``resource.name.startsWith`` bounds access to
    objects, but it cannot gate ``storage.objects.list``: for a list the resource named
    is the *bucket*, so a boundary with only that condition either blocks listing or
    exposes the whole bucket to it. ``objectListPrefix`` is what bounds the listing.
    Google's documentation is explicit that boundaries exist only for Cloud Storage, so
    nothing here generalises to another service.
    """
    if len(scope.prefixes) > _MAX_BOUNDARY_RULES:
        raise CredentialError(
            f"a Credential Access Boundary holds at most {_MAX_BOUNDARY_RULES} rules, "
            f"and this scope needs {len(scope.prefixes)}"
        )
    resource = f"//storage.googleapis.com/projects/_/buckets/{scope.bucket}"
    roles = [_GCS_READ_ROLE] if scope.read_only else [_GCS_READ_ROLE, _GCS_WRITE_ROLE]
    rules = []
    for prefix in scope.prefixes:
        # Every interpolation of a user-controlled name goes through cel_literal. This
        # single call is the difference between a scoped credential and a bucket-wide
        # one -- see CVE-2026-42811.
        path = cel_literal(f"{resource}/objects/{prefix}")
        rules.append(
            {
                "availableResource": resource,
                "availablePermissions": roles,
                "availabilityCondition": {
                    "title": "scoped-to-prefix",
                    "expression": f"resource.name.startsWith({path})",
                },
            }
        )
    return {
        "accessBoundary": {"accessBoundaryRules": rules},
        # Carried alongside because the listing bound is applied at the client rather
        # than in the boundary expression, and losing it in a port would silently widen
        # what a session can enumerate.
        "objectListPrefixes": list(scope.prefixes),
    }


# -- Azure -------------------------------------------------------------------------

#: A user delegation key lives at most seven days, and a SAS signed with one must expire
#: inside that window. Both bounds are Azure's, not ours.
MAX_DELEGATION_WINDOW = dt.timedelta(days=7)


def blob_sas_terms(
    scope: StorageScope, *, ttl: dt.timedelta, now: dt.datetime | None = None
) -> dict[str, Any]:
    """The terms of a user delegation SAS for ``scope``.

    A *user delegation* SAS, signed with a key derived from an Entra identity rather
    than with the storage account key: an account-key SAS cannot be revoked without
    rotating the key every other consumer also uses. Azure has no general analogue of a
    session policy, its own AWS comparison lists none, so per-request scoping exists for
    storage and nowhere else, and one directory prefix per SAS is the granularity.
    """
    if len(scope.prefixes) != 1:
        raise CredentialError(
            "a user delegation SAS scopes to one directory; vend one credential per "
            f"prefix (this scope has {len(scope.prefixes)})"
        )
    start = now or dt.datetime.now(dt.timezone.utc)
    seconds = _checked_ttl(ttl)
    expiry = start + dt.timedelta(seconds=seconds)
    if expiry - start > MAX_DELEGATION_WINDOW:  # pragma: no cover - MAX_TTL is smaller
        raise CredentialError("a user delegation SAS cannot outlive its delegation key")
    return {
        "container_name": scope.bucket,
        # `sr=d`: the signed resource is a directory, which is what makes the prefix the
        # boundary rather than a single blob.
        "resource": "d",
        "blob_name": scope.prefixes[0].rstrip("/"),
        "permission": "r" if scope.read_only else "racw",
        "start": start,
        "expiry": expiry,
    }


class GcsCredentialBroker:
    """Vends downscoped Cloud Storage tokens via a Credential Access Boundary."""

    def __init__(self, *, source_credentials: Any | None = None) -> None:
        self._source = source_credentials

    def vend(
        self, scope: StorageScope, *, ttl: dt.timedelta = DEFAULT_TTL
    ) -> VendedCredentials:
        """Downscope this deployment's credentials to ``scope`` and hand them over.

        A downscoped token has **no lifetime of its own** (it inherits the input token's
        expiry), so ``ttl`` bounds when the source is refreshed rather than the boundary
        itself. That is a property of the mechanism, not a shortcut here, and it is why
        a 15-minute TTL is not achievable on GCP the way it is on AWS.
        """
        boundary = gcs_access_boundary(scope)
        _checked_ttl(ttl)
        try:
            import google.auth
            from google.auth import downscoped
            from google.auth.transport import requests as google_requests
        except ImportError as exc:  # pragma: no cover - needs the extra absent
            raise CredentialError(
                "vending GCS credentials needs google-auth; install elbi-core[gcp]"
            ) from exc

        source = self._source
        if source is None:
            source, _project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        rules = [
            downscoped.AccessBoundaryRule(
                available_resource=rule["availableResource"],
                available_permissions=rule["availablePermissions"],
                availability_condition=downscoped.AvailabilityCondition(
                    expression=rule["availabilityCondition"]["expression"],
                    title=rule["availabilityCondition"]["title"],
                ),
            )
            for rule in boundary["accessBoundary"]["accessBoundaryRules"]
        ]
        credentials = downscoped.Credentials(
            source_credentials=source,
            credential_access_boundary=downscoped.CredentialAccessBoundary(rules=rules),
        )
        credentials.refresh(google_requests.Request())
        expiry = credentials.expiry
        if expiry is not None and expiry.tzinfo is None:
            # google-auth reports a naive UTC expiry; everything downstream compares
            # against an aware one.
            expiry = expiry.replace(tzinfo=dt.timezone.utc)
        return VendedCredentials(
            env={
                "GOOGLE_OAUTH_ACCESS_TOKEN": credentials.token,
                # The object-store client reads the prefix bound separately: a
                # resource.name condition cannot gate a list, so the listing bound
                # travels
                # alongside the token rather than inside it.
                "ELBI_STORAGE_PREFIXES": ",".join(boundary["objectListPrefixes"]),
            },
            expires_at=expiry or (dt.datetime.now(dt.timezone.utc) + ttl),
            scope=scope,
        )


class AzureCredentialBroker:
    """Vends a user delegation SAS over one directory of a container."""

    def __init__(
        self,
        account: str,
        *,
        credential: Any | None = None,
        account_url: str | None = None,
    ) -> None:
        self._account = account
        self._credential = credential
        self._account_url = account_url or f"https://{account}.blob.core.windows.net"

    def vend(
        self, scope: StorageScope, *, ttl: dt.timedelta = DEFAULT_TTL
    ) -> VendedCredentials:
        """Sign a SAS with an Entra-derived delegation key, not the account key.

        The account key is deliberately never used: a SAS signed with it cannot be
        revoked without rotating the key every other consumer of that account shares. A
        user delegation key can be revoked on its own, and Azure caps it at seven days:
        this asks for a window a little wider than the SAS, so clock skew cannot expire
        the key before the token it signed.
        """
        terms = blob_sas_terms(scope, ttl=ttl)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient, generate_blob_sas
        except ImportError as exc:  # pragma: no cover - needs the extra absent
            raise CredentialError(
                "vending Azure credentials needs azure-storage-blob and "
                "azure-identity; install elbi-core[azure]"
            ) from exc

        service = BlobServiceClient(
            self._account_url, credential=self._credential or DefaultAzureCredential()
        )
        delegation_key = service.get_user_delegation_key(
            key_start_time=terms["start"] - dt.timedelta(minutes=5),
            key_expiry_time=terms["expiry"] + dt.timedelta(minutes=5),
        )
        token = generate_blob_sas(
            account_name=self._account,
            container_name=terms["container_name"],
            blob_name=terms["blob_name"],
            user_delegation_key=delegation_key,
            permission=terms["permission"],
            start=terms["start"],
            expiry=terms["expiry"],
            # sr=d: the signed resource is a directory, which is what makes the prefix
            # the
            # boundary rather than one blob.
            sdd=len([p for p in terms["blob_name"].split("/") if p]),
        )
        return VendedCredentials(
            env={
                "AZURE_STORAGE_ACCOUNT_NAME": self._account,
                "AZURE_STORAGE_SAS_TOKEN": token,
            },
            expires_at=terms["expiry"],
            scope=scope,
        )


def broker_for(uri: str, **kwargs: Any) -> CredentialBroker:
    """The broker matching a storage URI's scheme.

    ``kwargs`` reach the chosen broker, so an AWS deployment passes ``role_arn`` and the
    others pass nothing. An unknown scheme is an error rather than a silent fallback to
    no credentials: a deployment that believes it is vending and is not would proxy
    every byte through the app and never find out why it was slow.
    """
    scheme = uri.split("://", 1)[0].lower()
    if scheme in ("s3", "s3a"):
        return AwsCredentialBroker(**kwargs)
    if scheme in ("gs", "gcs"):
        return GcsCredentialBroker(**kwargs)
    if scheme in ("abfs", "abfss", "az"):
        # abfss://container@account.dfs.core.windows.net/prefix: the account is in the
        # host, so a caller does not repeat what the URI already says.
        host = uri.split("://", 1)[-1].partition("/")[0]
        account = host.partition("@")[2].split(".")[0] or host.split(".")[0]
        return AzureCredentialBroker(kwargs.pop("account", account), **kwargs)
    raise CredentialError(f"no credential broker for storage scheme {scheme!r}")


def scope_for(
    uri: str, *, prefixes: Sequence[str], read_only: bool = True
) -> StorageScope:
    """The storage scope a session gets, from the warehouse URI and its prefixes."""
    remainder = uri.split("://", 1)[-1]
    bucket, _, root = remainder.partition("/")
    # abfss:// puts the container before an `@` and the account after it. Without this
    # the
    # container would be read as "container@account.dfs.core.windows.net", and the SAS
    # would be issued against a container that does not exist.
    if "@" in bucket:
        bucket = bucket.partition("@")[0]
    within = [f"{root.rstrip('/')}/{p.lstrip('/')}" if root else p for p in prefixes]
    return StorageScope(bucket=bucket, prefixes=tuple(within), read_only=read_only)
