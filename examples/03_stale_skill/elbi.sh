#!/usr/bin/env bash
# The same question as a tool: served fresh from whatever is in the CSV right now.
set -euo pipefail
cd "$(dirname "$0")"

elbi validate
elbi mcp --port 7878 > /tmp/elbi-mcp.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null' EXIT

# Wait for the server socket rather than a fixed sleep -- a first run in a fresh
# venv (no warm disk cache, nothing byte-compiled yet) can take 30s+ to come up;
# a warm one comes up in ~1s. 90s covers both.
ready=false
for _ in $(seq 1 450); do
    if (exec 3<>/dev/tcp/127.0.0.1/7878) 2>/dev/null; then
        ready=true
        break
    fi
    sleep 0.2
done
if [ "$ready" != true ]; then
    echo "elbi mcp never opened :7878 -- last output:" >&2
    tail -n 20 /tmp/elbi-mcp.log >&2
    exit 1
fi

# Resolve the interpreter from elbi's own shebang: elbi depends on mcp, so that
# Python is guaranteed to have it. Bare `python3` on PATH is not.
python_bin=$(sed -n '1s/^#!//p' "$(command -v elbi)")
if [ -z "$python_bin" ] || ! "$python_bin" -c "import mcp" >/dev/null 2>&1; then
    echo "no Python with the 'mcp' client library found next to the elbi CLI." >&2
    echo "from a repo checkout, try: uv run ./elbi.sh" >&2
    exit 1
fi

"$python_bin" call_overlay.py
