#!/bin/sh
# Container entrypoint: ensure a writable project exists, then serve on all interfaces.
#
# The app serves an elbi *project* (datasets, derivations, config). On first run
# against an empty or freshly-mounted volume, seed the bundled starter project so the
# container boots without any host setup; mount your own project over the data dir to
# override it. The durable store is external whenever DB_URI points at Postgres: this
# only concerns the project files and their local scratch/cache under .elbi.
set -eu

# Assemble DB_URI from discrete parts when it is not supplied directly. This lets the
# database password be injected on its own from a rotated secret (e.g. AWS RDS managed
# credentials) while the non-secret host/port/name/user come as plain config. The
# password must be URL-safe (RDS-managed passwords exclude URI-breaking characters).
#
# DB_PASSWORD_FILE is honoured here because the app's own `_FILE` convention cannot reach
# this: the application resolves `<NAME>_FILE` when it reads a variable, and by then the URI
# has already been assembled from the shell. Without this the documented convention would
# silently produce a URI with an empty password -- which fails as an authentication error
# well away from its cause. A file that is set but unreadable is fatal rather than empty,
# for the same reason.
if [ -z "${DB_URI:-}" ] && [ -n "${DB_HOST:-}" ]; then
    # `DB_PASSWORD_FILE` is resolved here because the application's own `_FILE` handling
    # happens when it *reads* a variable, and by then this URI is already built. The rules
    # are the app's rules, so one convention behaves identically wherever a secret is
    # consumed -- and they are the rules the Docker official images' `file_env` established
    # for this convention: both failure modes are fatal.
    if [ -n "${DB_PASSWORD_FILE:-}" ]; then
        if [ -n "${DB_PASSWORD:-}" ]; then
            echo "error: DB_PASSWORD and DB_PASSWORD_FILE are both set, but they are" >&2
            echo "       exclusive. Supply the password either directly or as a file." >&2
            exit 1
        fi
        if [ ! -r "${DB_PASSWORD_FILE}" ]; then
            echo "error: DB_PASSWORD_FILE=${DB_PASSWORD_FILE} could not be read." >&2
            echo "       Refusing to fall back: connecting with an empty password would" >&2
            echo "       surface a missing secret mount as an authentication failure." >&2
            exit 1
        fi
        # Surrounding whitespace only -- `tr -d` would eat spaces *inside* a password and
        # turn a valid credential into a silent authentication failure.
        DB_PASSWORD="$(
            sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "${DB_PASSWORD_FILE}"
        )"
    fi
    export DB_URI="postgres://${DB_USER:-postgres}:${DB_PASSWORD:-}@${DB_HOST}:${DB_PORT:-5432}/${DB_NAME:-postgres}"
fi

DATA_DIR="${ELBI_PROJECT_DIR:-/data}"

# Where the project *comes from*, which decides what a missing project means.
#
#   bundled  nobody is delivering one, so seed the starter and boot (the laptop case)
#   image    it was baked into this image
#   git      a git-sync sidecar populates it before this container starts
#
# The distinction exists to stop one specific bad outcome. When something else is meant to
# deliver the project and has not, seeding the starter would leave the app running happily
# on example data under the customer's project name: wrong answers, confidently served,
# with nothing in the logs that reads as a failure. Anything other than `bundled` refuses
# to start instead.
PROJECT_SOURCE="${ELBI_PROJECT_SOURCE:-bundled}"

# State is separated from source so the source can be read-only: baked into the image, or a
# worktree git-sync rewrites on every commit. Defaults inside the project, which is what a
# laptop wants and what every existing deployment already has.
STATE_DIR="${ELBI_STATE_DIR:-${DATA_DIR}/.elbi}"

if [ ! -f "${DATA_DIR}/elbi.yaml" ]; then
    if [ "${PROJECT_SOURCE}" != "bundled" ]; then
        echo "error: no elbi.yaml at ${DATA_DIR}, and this deployment expects" >&2
        echo "       its project from '${PROJECT_SOURCE}'. Refusing to start on the" >&2
        echo "       starter project, which would serve example data as if it were yours." >&2
        if [ "${PROJECT_SOURCE}" = "git" ]; then
            echo "       Check the git-sync init container's logs: the clone is what" >&2
            echo "       populates this directory before the app runs." >&2
        fi
        exit 1
    fi
    echo "No project at ${DATA_DIR}; seeding the starter project."
    mkdir -p "${DATA_DIR}"
    # -n: never clobber files an operator already placed in a mounted volume.
    cp -Rn /opt/elbi/example-project/. "${DATA_DIR}/" 2>/dev/null || true
fi
mkdir -p "${STATE_DIR}"

# `serve` is named explicitly: the CLI has more than one command (migrate, diagnostics), so
# there is no implicit default and its options are not accepted at the top level.
exec elbi-app serve \
    --directory "${DATA_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT:-7700}" \
    --no-browser
