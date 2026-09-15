#!/usr/bin/env bash
# The file-backed agent: read the skill, act on what it says.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== agent reading skills/risk.md ==="
echo
sed -n '1,20p' skills/risk.md
echo
echo "=== action it would take ==="
echo "BROKER: cut TTD 3%   (source: skills/risk.md, 2026-08-06 close)"
echo
echo "=== last bar in fixtures/returns.csv ==="
tail -n 1 fixtures/returns.csv
