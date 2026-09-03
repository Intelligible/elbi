# Upgrading

## Finding out whether there is anything to upgrade to

```bash
elbi update
```

It asks PyPI what the newest release is, compares it to what you have, and prints the
command that upgrades *your* install, which differs depending on how Elbi got onto
the machine. It works that out rather than guessing.

```
! A new release of elbi is available: 0.1.0 -> 0.2.0

→ To upgrade:
    uv tool upgrade elbi
```

Nothing checks on your behalf. There is no check when the app starts, none on a timer,
and none folded into another command. Other commands do use the network (`sync`,
`pull` and `upload` talk to your running app), but this is the only one that contacts
anything outside your own infrastructure, and only when you type it. So
[the privacy page](privacy.md) stays true as written.

The request is a GET for `https://pypi.org/pypi/elbi/json`, a public file, carrying a
User-Agent that names the tool and its version. No identifier, no counter, nothing about
the machine. If you would rather not make it at all, don't run the command; the version you
have is in `elbi --version` and in the app under **Settings → About**.

`--quiet` prints nothing when you are current and exits non-zero when you are not, which
is the form to use in a script:

```bash
elbi update --quiet || echo "time to upgrade"
```

Where the project publishes release notes, the check links them, so you can see what
changed before deciding.

## Doing the upgrade

`elbi update` prints the command by default. `elbi update --apply` runs it.

`--apply` hands the process over to the installer with `exec` rather than starting it
alongside. The upgrade rewrites the environment this interpreter is running out of, and
a Python process that keeps going while its own files are replaced can fail on the next
import it happens to make; replacing the process image means nothing of Elbi is running
by the time anything changes, and the installer's exit status becomes the command's.

It refuses two cases rather than guessing. A container upgrades with two commands joined
by `&&`, which needs a shell; that is printed for you to run, not executed on your
behalf. And Kubernetes has no single command at all, so it says so.

| How you installed it | How you upgrade it |
| --- | --- |
| `uv tool install elbi` | `uv tool upgrade elbi` |
| `pipx install elbi` | `pipx upgrade elbi` |
| `pip install elbi` in a virtualenv | `pip install --upgrade elbi` with that environment active |
| `pip install --user elbi` | `pip install --user --upgrade elbi` |
| The container image | Change the tag, then `docker compose pull && docker compose up -d` |

If you installed `elbi-cli` on its own (the MCP-only install, without the chat
app), substitute that name throughout. `elbi update` already does; it upgrades whichever
distribution is actually present.

## Containers

Pin `APP_VERSION` to a release tag for anything you intend to keep, because `latest`
moves with `main`. Change the tag, then recreate:

```bash
docker compose pull && docker compose up -d
```

The image sets `ELBI_INSTALL=docker`, so `elbi update` inside the container prints the
compose command rather than a `pip` one that would not survive the next restart.

For Kubernetes, change the image tag in your values file and roll the deployment.
`elbi update` recognises a pod (every one has `KUBERNETES_SERVICE_HOST` set) and says
that rather than printing a compose command that could not work there. Run one replica
through the migration before scaling back up.

## What happens to your data

The database schema migrates forward on start. Take a backup first for anything you care
about, because the migration is one-way and downgrading afterwards will not undo it.

Nothing else moves. Your warehouse tables, your derivations and their cached results,
and your project files are untouched by an upgrade; they live outside the package.

## Pre-releases

`elbi update` ignores pre-releases unless the version you are running is itself a
pre-release, so a stable install is never told to move onto an alpha. `--pre` asks for
them anyway. Releases whose
files have all been yanked are skipped too, because upgrading onto one is exactly what
yanking is meant to prevent.
