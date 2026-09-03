#!/usr/bin/env bash
# Make room on a runner for an image build that needs most of the disk.
#
# The numbers, measured on main rather than guessed. The first version of this reclaim
# left the runner 46 GB of 72 GB free, and the build finished at 98% -- 2.1 GB left on the
# arm64 leg, which passes, and nothing left on the amd64 leg, which does not: it dies
# mid-build with no failed step, because the runner fills its own diagnostic log and the
# worker throws `No space left on device` before it can upload anything. All that survives
# is an annotation, which is the only reason it is diagnosable at all. The second set
# below is what turned that margin from noise into a few gigabytes.
#
# So the build needs ~44 GB and both legs sit at the edge of what the reclaim leaves. Two
# categories the earlier version did not touch are worth about 9 GB together, which is the
# difference between 2 GB of headroom and 11 GB:
#
#   * swap, 4 GB of a disk-backed file that a build never benefits from
#   * the preinstalled browsers, cloud CLIs and Mono, ~5 GB of apt packages
#
# This is the same set the community `free-disk-space` action removes, implemented here
# rather than taken as a dependency: every action in this repository is pinned by digest
# and the image is signed and attested, so adding a third-party step whose whole job is
# `sudo rm -rf` as root is a supply-chain surface worth not adding for 9 GB.
#
# Which is also what comparable projects do. jupyter/docker-stacks writes its own
# composite action over the same paths; LocalAI uses the community action and then adds
# its own apt purges on top, because for an image this size the action alone is not
# enough. Neither treats it as solved by a dependency.
#
# Relocating Docker to a second disk is the other documented remedy and does not apply:
# these runners mount no /mnt, so there is nowhere to move it to. Measured, not assumed --
# `df -h / /mnt` reports the root filesystem twice.
set -euo pipefail

report() {
  echo "--- disk $1 ---"
  df -h / | tail -1
  # Also to the job summary. When this reclaim is not enough the runner fills its own
  # diagnostic log and dies before uploading any, so the log that would say how close it
  # came is exactly the one that does not survive. A summary does.
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "\`disk $1\`: $(df -h / | tail -1 | awk '{print $4}') available" \
      >> "$GITHUB_STEP_SUMMARY"
  fi
}

report before

# Toolchains an image build never uses. The Android SDK alone is the largest single item
# on the runner. Globs are quoted where the version in the path changes between runner
# images, so a rename leaves the entry inert rather than deleting a parent directory.
sudo rm -rf \
  /usr/share/dotnet \
  /usr/local/lib/android \
  /opt/ghc \
  /usr/local/share/boost \
  /usr/local/share/powershell \
  /usr/local/lib/node_modules \
  /usr/share/swift \
  /usr/local/.ghcup \
  /usr/share/miniconda \
  /opt/hostedtoolcache/CodeQL \
  "${AGENT_TOOLSDIRECTORY:-}"

# A second set, added because the first left the amd64 leg finishing at 100% of the disk
# while arm64 finished at 98%: the same build, a few hundred megabytes apart, so which
# one passed came down to noise. None of these is large on its own and together they are
# a few gigabytes, which is the difference between a margin and a coin toss.
sudo rm -rf \
  /usr/lib/jvm \
  /opt/pipx \
  /usr/share/kotlinc \
  /usr/local/lib/heroku \
  /usr/local/share/chromium \
  /usr/local/julia* \
  /imagegeneration

# Swap is a file on the very disk the build is short of, and a build that starts swapping
# has already lost. Reclaiming it is pure gain here.
sudo swapoff -a || true
sudo rm -f /mnt/swapfile /swapfile

# Browsers, cloud CLIs and Mono. Listed explicitly rather than by pattern so a package
# this build does need cannot be swept up by accident. `|| true` because the set present
# on the runner image changes and a missing package must not fail the job.
sudo apt-get -qq purge -y --autoremove \
  azure-cli \
  google-cloud-cli \
  google-chrome-stable \
  microsoft-edge-stable \
  firefox \
  mono-devel \
  >/dev/null 2>&1 || true
sudo apt-get -qq clean
sudo rm -rf /var/lib/apt/lists/*

sudo docker image prune --all --force >/dev/null

report after
