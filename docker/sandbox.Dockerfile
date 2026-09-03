# The data-science sandbox image for `sandbox: docker`.
#
# Build it once with `just sandbox-image` (tags it `elbi-sandbox:latest`), then
# point a project at it with `sandbox_image: elbi-sandbox` in elbi.yaml.
# It bakes in a build toolchain and the common analysis stack so the exploration agent
# does not spend turns installing them (or hit missing-compiler build failures for
# source-only packages, or the OpenMP gap that stops lightgbm). A certified derivation
# still provisions any extra declared dependency into its own hermetic volume.
FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

# build-essential provides the C/C++ toolchain that source-only wheels (e.g. interpret's
# aplr) need to compile; libgomp1 is the OpenMP runtime lightgbm and xgboost link.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The stack an analysis agent reaches for. Kept unpinned so a rebuild picks up current
# stable versions; pin here if a project needs reproducible sandbox versions.
RUN pip install --no-cache-dir \
    numpy pandas scipy scikit-learn statsmodels \
    xgboost lightgbm interpret
