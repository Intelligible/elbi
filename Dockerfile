# syntax=docker/dockerfile:1.9
#
# Multi-stage build for the elbi chat app:
#   1. web: build the React/Vite single-page app
#   2. builder: install the uv workspace (all runtime extras) into a venv
#   3. runtime: a slim, non-root image with just the venv, source, and SPA
#
# The store is external via DB_URI (Postgres) in production; the image bundles a starter
# project so it boots with no host setup, and a mounted volume at /data overrides it.

############################  Stage 1: frontend (Vite)  ########################
FROM node:25-slim@sha256:81db02c4b671288a03915da9534dbd54f96d0e7c24d80ccc54f5b36b2e684370 AS web
WORKDIR /web
# Install deps against the lockfile first, so this layer caches across source edits.
COPY packages/elbi/web/package.json packages/elbi/web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY packages/elbi/web/ ./
# Emit the build to a known path (overrides the config's in-source outDir).
RUN npm run build -- --outDir /web/dist --emptyOutDir

############################  Stage 2: python builder  ########################
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# 2a. Dependency layer: copy the lockfile + every workspace member's manifest, then
#     install third-party deps only (cached until a manifest or the lock changes).
COPY pyproject.toml uv.lock ./
COPY packages/elbi-core/pyproject.toml        packages/elbi-core/
COPY packages/elbi-agent/pyproject.toml  packages/elbi-agent/
COPY packages/elbi/pyproject.toml    packages/elbi/
COPY packages/elbi-cli/pyproject.toml    packages/elbi-cli/
# --all-packages installs every workspace member; the app ships all its features in its
# base deps (connectors, ML, tracing), so no app extras are needed. --all-extras still
# enables the core `elbi` SDK's extras: a bare --all-extras only sees the virtual root's
# extras and would omit the members'.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-workspace --no-dev --all-packages --all-extras

# 2b. Full source + the built SPA, then install the workspace packages themselves.
COPY . /app
COPY --from=web /web/dist \
    /app/packages/elbi/src/elbi/web/dist
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --all-packages --all-extras

############################  Stage 3: runtime  ###############################
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
LABEL org.opencontainers.image.source="https://github.com/Intelligible/elbi" \
      org.opencontainers.image.description="Verified-analysis chat app" \
      org.opencontainers.image.licenses="Apache-2.0"

# OpenMP runtime the ML wheels (lightgbm/xgboost/scikit-learn) dlopen at import; the
# slim base omits it. curl and ca-certificates are for the kubectl fetch below and are
# kept: the entrypoint and the image's own health tooling use them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# kubectl, because the Kubernetes runner starts a kernel by shelling out to it:
# `kubectl attach` is the only way to reach a pod's stdin, which is what the kernel
# protocol speaks. Without it `SANDBOX_BACKEND=kubernetes` installs cleanly and then
# fails on the first cell with "kubectl executable not found", which is a packaging
# defect wearing a configuration error's clothes.
#
# Pinned and checksum-verified rather than tracking `stable.txt`, so a rebuild produces
# the same image and a compromised CDN cannot substitute a binary. kubectl supports one
# minor version of skew in either direction, so this covers clusters from 1.35 to 1.37.
ARG KUBECTL_VERSION=v1.36.3
ARG TARGETARCH
RUN curl -fsSLO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
    && curl -fsSLO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl.sha256" \
    && echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check --status \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm -f kubectl kubectl.sha256 \
    && kubectl version --client=true --output=yaml > /dev/null

# Non-root runtime user (uid 1000 matches the Helm chart's securityContext default).
#
# The primary group is root (GID 0), not a private one. That is what makes the image work
# under an arbitrary UID: OpenShift's restricted-v2 SCC ignores the USER directive and runs
# the container as a random UID from the project's range, but always in GID 0. Owning the
# writable paths by group 0 with the group bits mirroring owner (chmod g=u, applied below)
# means that random UID can still write. A private group would leave it unable to.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid 0 --create-home app

WORKDIR /app

# The editable install points the venv back at /app source, so ship both.
COPY --from=builder --chown=app:0 /app/.venv /app/.venv
COPY --from=builder --chown=app:0 /app       /app

# The starter project (seeded into /data on first run) and the entrypoint.
COPY --chown=app:0 deploy/example-project /opt/elbi/example-project
COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /data \
    # Every path the app writes to: the project dir, the source tree (editable install
    # scratch), the starter project, and HOME: the default STORAGE_URI lives under
    # ~/.elbi, so an unwritable HOME breaks the warehouse on a bare `docker run`.
    && chown -R app:0 /data /app /opt/elbi /home/app \
    && chmod -R g=u  /data /app /opt/elbi /home/app

# HOME is set explicitly because an arbitrary UID has no /etc/passwd entry, so
# Path.home() would fall back to "/", and the default STORAGE_URI lives under
# ~/.elbi, which would then be an unwritable path at the filesystem root.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    PORT=7700 \
    # Declares how this copy was installed, so `elbi update` prints the command that
    # actually applies here instead of inferring one from the filesystem. A container
    # can be built any number of ways and a wrong guess names a command that does not
    # work; saying so directly costs one line.
    ELBI_INSTALL=docker
# Numeric, not `app`. Kubernetes cannot verify a name against `runAsNonRoot` -- it reads
# this field and refuses to start a pod whose user it cannot prove is not root. The uid
# is the one created above.
USER 1000
VOLUME /data
EXPOSE 7700

# Exec form, so the probe is one process rather than a shell that then runs python --
# it fires every 30 seconds for the life of the container.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7700/health/ready').status==200 else 1)"]

ENTRYPOINT ["docker-entrypoint.sh"]
