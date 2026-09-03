# Compute

Where a notebook's code runs, and how big it may be.

Infrastructure defines the menu; you pick from it. Profiles live in the project file or
the environment, and the app shows them without offering to edit them, because a size you
can change from inside the app is a size the machine underneath may not have.

## Runners

| Runner | Where a kernel runs | Sizing | Isolation |
| --- | --- | --- | --- |
| `subprocess` | beside the app, in its own process | none | a scrubbed subprocess with the network denied |
| `docker` | a container on the same host | cgroup | container, filesystem, and an egress proxy |
| `kubernetes` | one pod per session | cgroup, scheduled | pod, plus the cluster's `runtimeClassName` |

`subprocess` is the laptop and single-machine tier. It is deliberately unsized: the only
mechanism available to a process, `RLIMIT_AS`, caps virtual address space rather than
resident memory and counts file mappings, so it misfires in ordinary numpy and Arrow code
long before real memory is exhausted. A limit that fires spuriously is worse than none. A
cgroup is the only honest place to cap memory, so the deployed tiers have one and this
one does not.

`kubernetes` is the one a real deployment should use. The reason is the self-hosting one:
your cluster already has capacity, an autoscaler, GPU node pools and, through
`runtimeClassName`, gVisor or Kata isolation, so a sandbox can be as strong as your
platform allows without this product shipping a hypervisor.

Select it with `SANDBOX_BACKEND=kubernetes`, and name the namespace kernels run in with
`COMPUTE_NAMESPACE` (optional; it is also where a `ResourceQuota` would go).

Agent-written derivation candidates run on this runner too, as a Job per candidate:
`restartPolicy: Never`, `backoffLimit: 0`, a TTL for cleanup. Kubernetes' own guidance
prefers a Job to a bare Pod even for a single pod, because a bare Pod on a failed node is
simply gone; the zero retries are ours, because a candidate is deterministic and a retry
reproduces the same failure at twice the cost. A candidate's inputs travel in the pod's
environment, which bounds them at 96 KiB; past that the run is refused with a message
saying to bind the data as a warehouse table instead of passing it.

Kernels attach over the Kubernetes API, so the app needs pod verbs in that namespace: a
Role granting exactly `pods`, `pods/attach` and `pods/exec`, and nothing else. The kernel
pod itself gets no identity at all: no service-account token, no
service environment variables, every capability dropped, a read-only root filesystem and
a non-root user. The grant belongs to the app, which is trusted; not to the sandbox,
which is not.

Two things follow from the app reaching the API through `kubectl` rather than a client
library, and a deployment that misses either is broken in a way it only discovers when
somebody runs a cell. The app image ships `kubectl`, because `kubectl attach` is the only
way to reach a pod's stdin, and stdin is what the kernel protocol speaks. And the app's pod
must have its service-account token mounted: `kubectl` authenticates in-cluster with
exactly that token, so a Role granted to a service account that cannot present itself is
no grant at all.

## Profiles

A profile is a named shape with its limits and its audience:

```yaml
compute:
  profiles:
    - name: small
      cpu: "2"
      memory: 4Gi
    - name: large
      cpu: "8"
      memory: 32Gi
      idleTimeout: 3600
    - name: gpu
      cpu: "8"
      memory: 32Gi
      gpu: 1
      gpuType: nvidia-l4
      runtimeClass: gvisor
      allowedGroups: [ml]
      egress: none
  defaultProfile: small
```

Quantities are Kubernetes quantities (`"2"` or `"500m"` for CPU, `8Gi` for memory)
everywhere, including on the docker runner, which converts. Note the trap Kubernetes
itself carries: a lowercase `m` on memory means *millibytes*, so `memory: 512m` is a
request for half a byte. It is rejected rather than honoured.

| Field | Meaning |
| --- | --- |
| `cpu`, `memory` | Kubernetes quantities. Requests equal limits, so a kernel gets the Guaranteed QoS class and is not evicted to make room. |
| `gpu`, `gpuType` | Whole GPUs, and the node label selecting the accelerator. |
| `image` | Overrides the deployment's kernel image for this profile, so a GPU profile can carry a CUDA userland without every profile paying for it. |
| `idleTimeout` | Seconds with no execution and no attached client before the kernel is reaped. Default 30 minutes, and the **cap** a notebook's own setting is clamped to. |
| `maxRuntime` | Seconds one cell may run. A separate limit, because a forgotten kernel and a runaway cell are different failures. |
| `egress` | `full`, `none`, or a host allowlist. Intersected with the project's policy; never widens it. |
| `allowedGroups` | Org roles whose members may select this profile. Empty means everyone. |
| `spot` | Interruptible capacity: cheap, and an eviction loses the kernel. Right for batch, wrong for an interactive default. |
| `runtimeClass` | `gvisor` on GKE Sandbox, `kata-vm-isolation` on AKS. |
| `warmPoolSize` | Kernels kept started ahead of demand. Zero by default; the customer pays for idle capacity. |
| `shared` | Put every notebook on this profile into **one** kernel. See [Shared kernels](#shared-kernels). |
| `hidden` | Fields to leave out of what the app shows. Changes the display, never the enforcement. |

Set none of this and you get `small`, `medium` and `large` at sensible defaults.

A notebook's profile belongs to the notebook rather than to whoever opens it, so two
collaborators do not get two different machines. Anyone who can edit the notebook can
change it, and the change is audited.

A notebook may also shorten its own idle timeout, through its metadata, and never extend
it: holding a kernel is a cost someone else pays, so the profile is the admin cap:

```json
{"metadata": {"idle_timeout": 300}}
```

### When a profile changes

Databricks documents the failure mode precisely: after a policy changes, compute created
under it "aren't automatically updated". So an admin who tightens a memory limit believes
they have and has not.

Each profile carries a version derived from its own content, a running kernel records the
version it started with, and `GET /api/compute/notebooks/{id}` reports the difference:

```json
{
  "profile": {"name": "small", "memory": "1Gi"},
  "startedWith": {"name": "small", "memory": "4Gi"},
  "drift": "running on small as it was defined at version a1b2c3…; restart the kernel to pick it up."
}
```

Restarting is what applies it. A profile edit does not kill live kernels, because
resizing a profile should not destroy the state of everyone currently working.

### Warm pools

A profile can keep kernels started ahead of demand, so opening a notebook does not wait
for a pod:

```yaml
compute:
  profiles:
    - name: small
      warmPoolSize: 2
```

Zero by default, and it should usually stay there: you pay for that idle capacity
whether or not a notebook opens. The pool is keyed by the profile *as defined at that
moment* and by the exact dependency set, so a resized profile never hands out a kernel
built to the old size and a notebook with its own dependencies never gets one warmed
without them. A per-notebook scratch workspace disqualifies a session from the pool,
because a pooled kernel is started before any notebook is known.

### Shared kernels

`shared: true` puts every notebook on a profile into one interpreter. It is opt-in
because whatever one notebook leaves in the namespace is what the next one sees, and a
shared kernel cannot carry per-notebook credentials. Databricks says the same of their
equivalent, "data or internal credentials provisioned to that environment might be
accessible to any code running within that environment", and has since made that mode
legacy and off by default for new accounts.

The case it is good for is a cheap scratch tier, where a process per notebook is not
worth paying for and nothing sensitive is in the room.

## Environments

A notebook's dependencies are resolved to a lock and content-addressed by
`sha256(image + packages)`, so the same set is provisioned once and reused across runs
and sessions. What changes per runner is only what the digest names: a docker volume, or
a Kubernetes volume populated by an init container.

That init container checks for a completion sentinel first and writes it last, so an
interrupted install is never mistaken for a finished one. With a `ReadWriteMany` storage
class the install happens once per dependency set and is shared:

```yaml
compute:
  depsStorageClass: efs-sc   # must support ReadWriteMany
```

Without one, each session installs into its own scratch. Slower, and correct on the block
storage most clusters actually have, which a default requiring RWX would not be.

## Egress

A profile's network policy is enforced by the cluster, not by the sandbox, so code
running in a notebook cannot lift its own restriction. The runner labels each pod and the
a deployment supplies the network policies that select on it: `none` denies every
outbound connection,
`restricted` permits DNS and the ranges you name, `full` is unselected and therefore
unrestricted.

One honest limitation: a NetworkPolicy selects on IP and label, never on hostname, so a
profile's host allowlist cannot be expressed there as written. State the equivalent
ranges:

```yaml
compute:
  networkPolicy:
    allowedCIDRs: ["10.0.0.0/8"]
```

## Where SQL runs

Explore, metrics, dashboards and a notebook's `sql()` all funnel through one query path.
DuckDB is an *in-process* engine, so executing there means the engine and the request
handlers compete for the same pages, and a query that runs out of memory ends the web
server rather than the query. DuckDB's own community is direct about it: embedded in a
production server, "an out-of-memory crash takes down the entire service."

That is the same failure this design fixed for notebooks, so it gets the same answer.

```yaml
compute:
  queryRunner: worker      # or `inprocess`; empty follows the kernel runner
  queryPoolSize: 2
  queryTimeoutSeconds: 300
```

`worker` runs queries in a small pool of persistent child processes that own DuckDB. A
query's memory belongs to a worker; one that exhausts it dies and is replaced, and the app
never notices. The pool is small because each worker is a whole DuckDB, and larger than one
because a dashboard with several panels should not queue behind itself.

`inprocess` executes in the app. That is right for a laptop, where a worker pool buys
isolation nobody needs, and wrong for anything serving more than one person. Left empty,
the setting follows the kernel runner: `subprocess` gets in-process, anything else gets
workers.

DuckDB 1.5.2 added a native client-server protocol, which is the eventual shape for a
query service addressed over the network: a Deployment scaled on its own. The property
that matters first is the boundary, not where the boundary lives.

## Cost

Every finished session is attributed to `(user, notebook, profile, duration)` from the
first one, because usage history cannot be backfilled; Databricks is explicit that
missing tags "can't be added to past events".

```yaml
compute:
  maxCostPerHour: "5"      # refuse to offer a profile estimated above this
  spendLimit: "500"        # alert when the window's attributed spend passes this
  costRates: '{"cpu_core_hour": 0.04, "memory_gib_hour": 0.005, "gpu_hour": 1.0}'
```

`maxCostPerHour` is checked when the menu loads, so a profile nobody may launch is never
offered. `spendLimit` is an **alert, not a cap**: sessions keep running. Databricks is
candid that its own compute spend limits are notification-only and that they do "not
proactively terminate resources to maintain the limit", and ending someone's session to
recover the overage destroys work to save cents.

The rates are an estimate for comparing profiles and enforcing a ceiling, not a bill.
Databricks does the same thing with a synthetic unit rather than currency. The defaults
are rough on-demand list prices and are wrong for anyone on reserved capacity.

`GET /api/compute/usage` reports the window, split by **interactive** and **scheduled**.
Those mean different things: batch is work somebody scheduled, interactive is work
somebody is doing, and only the second is worth chasing when it sits idle. Databricks
bills them to different SKUs and does "not recommend" running production jobs on
all-purpose compute.

### What a pushdown cost

A cell records every query it sent to the warehouse (the statement, its duration, rows
returned, whether the result was truncated, the error if it failed) and shows them under
the cell. The fields follow Databricks' query history (statement, duration, rows
produced), minus what a single-node engine cannot honestly report such as bytes scanned.

This exists because the scale story *is* pushdown, and a claim about where work happens
should be checkable. A failed query is kept with its duration rather than dropped: one
that failed after twelve seconds is the one worth seeing.

## Storage credentials

A kernel fetches data on demand, so it needs to reach the warehouse. Configure a
credential broker and each session gets a short-lived credential scoped to the prefixes
it is entitled to; configure none and data reaches a cell through the app instead, and
the sandbox holds no credential at all.

Scoping is the whole trade, and it is a place worth being careful: Apache Polaris, the
reference implementation of this pattern, shipped two critical CVEs in one release from
user-controlled names landing in a policy language. A quote in a table name closed a CEL
literal and made a prefix restriction bucket-wide; a `*` in a table name became an IAM
wildcard. Both defences here are structural: every value entering a CEL expression is
escaped, and every prefix is normalised to end at a separator, refuses `..`, and refuses
the wildcard characters IAM offers no way to escape.

On AWS, set `SANDBOX_ROLE_ARN` to a role the app may assume and whose own policy grants
read of the warehouse bucket; the Terraform module creates one with
`create_sandbox_role = true`. The warehouse has to live under a prefix
(`s3://bucket/warehouse`), because a bucket root has nothing to scope to and the app
refuses to vend rather than issue a credential covering the whole bucket.

A session's credential is scoped to the warehouse root and read-only, not to the tables
one notebook has opened. A notebook can query any registered table, so a per-table
credential would be reminted on every query and would still cover everything a cell could
ask for by the end of the session, narrowing to the warehouse, to reads, for fifteen
minutes is the bound that actually holds.

All three clouds vend, behind their own extras: `elbi-core[aws]`, `[gcp]`,
`[azure]`. AWS assumes a role with an inline session policy; GCS downscopes the pod's own
token with a Credential Access Boundary; Azure signs a user delegation SAS over one
directory. On GCS and Azure there is no role to name, because the broker downscopes or delegates
from the identity the pod already has, so vending is switched on with
`SANDBOX_VEND_CREDENTIALS=true` rather than by supplying an ARN, and it is off by default
because handing a sandbox a storage credential is a decision.

Two mechanism differences worth knowing. A downscoped GCS token has **no lifetime of its
own**: it inherits the input token's expiry, so a short TTL comes from minting a fresh
input token rather than asking for a short boundary. And an Azure SAS is capped at seven
days by its delegation key, which the broker brackets slightly wider than the SAS so clock
skew cannot expire the key before the token it signed.

## What this does not do

Stated plainly, because each of these is a reasonable thing to expect.

**No Spark.** There is no `spark` object, and there is deliberately no shim over one. A
`spark.read` that works beside a `spark.sql` window function that silently differs is a
worse outcome than an honest absence: the partiality gets discovered at the worst
possible moment. Push work down with `sql(...)`, which runs against the warehouse where
the engine and the data already are. If you have real Spark assets, the intended path is
to keep running them on the cluster you have.

**No multi-GPU, no distributed training.** A GPU profile is one GPU on one node. This is
the mainstream position rather than a compromise: Databricks' AI Runtime accelerators
"provision a single node", Hex ships two profiles of one GPU each, and on Databricks a
GPU forces dedicated access mode, so a GPU independently implies a single-user machine.

**No autoscaling within a session.** A Databricks cluster scales workers between a min
and a max while running. A single Python kernel has nothing to scale *out* to; resizing
means restarting, which loses state. The honest equivalents are picking a bigger profile
and pushing work down.

**No init scripts.** Databricks supports them and now actively discourages them:
"Databricks recommends using compute policies instead of init scripts to install
libraries", and they do not run on serverless at all. The declarative dependency-plus-lock
design is what they are steering people towards, so the absence of an arbitrary pre-start
shell hook is the feature.

**No snapshot and restore.** A reaped kernel restarts cold.

**SQL does not have a separately scaled service.** Query work runs on its own processes
(see [Where SQL runs](#where-sql-runs)), which is the isolation boundary that matters, but
not on compute you scale independently of the app the way a Databricks SQL warehouse is.
Adding panels to a dashboard raises load on the app's own pod rather than on something you
size for it.
