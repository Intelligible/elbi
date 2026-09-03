"""Compute profiles: the sized, governed shapes a sandbox may run as.

A profile is the unit an operator defines and a user picks from: cpu, memory, gpu,
image, timeouts, egress, and who may use it. Infrastructure defines the menu, a person
chooses from it, which is the split Databricks arrived at with compute policies and the
one this product already uses for model ARNs.

Two things here are deliberate rather than incidental:

*Quantities are Kubernetes quantities.* ``cpu: "2"`` or ``"500m"``, ``memory: "8Gi"``.
The flagship runner is Kubernetes, so its notation is the one that survives the trip
unchanged; the docker backend converts, rather than every other layer converting.

*A profile has a version, derived from its own content.* Databricks documents the
failure mode plainly (after a policy changes, "the compute resources created using that
policy aren't automatically updated"), so an admin who tightens a memory limit believes
they have and has not. A content-addressed version makes drift visible: a session
records what it started with, and a profile that no longer matches is out of compliance.
"""

from .credentials import (
    DEFAULT_TTL,
    MAX_TTL,
    AwsCredentialBroker,
    AzureCredentialBroker,
    CredentialBroker,
    CredentialError,
    GcsCredentialBroker,
    StorageScope,
    VendedCredentials,
    blob_sas_terms,
    broker_for,
    cel_literal,
    gcs_access_boundary,
    s3_session_policy,
    scope_for,
    storage_prefix,
)
from .profiles import (
    ComputeProfile,
    ComputeProfileError,
    ComputeProfiles,
    CostRates,
    docker_resource_args,
    kubernetes_resources,
    parse_cpu,
    parse_memory,
)

__all__ = [
    "DEFAULT_TTL",
    "MAX_TTL",
    "AwsCredentialBroker",
    "AzureCredentialBroker",
    "ComputeProfile",
    "ComputeProfileError",
    "ComputeProfiles",
    "CostRates",
    "CredentialBroker",
    "CredentialError",
    "GcsCredentialBroker",
    "StorageScope",
    "VendedCredentials",
    "blob_sas_terms",
    "broker_for",
    "cel_literal",
    "docker_resource_args",
    "gcs_access_boundary",
    "kubernetes_resources",
    "parse_cpu",
    "parse_memory",
    "s3_session_policy",
    "scope_for",
    "storage_prefix",
]
