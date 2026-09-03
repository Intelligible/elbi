"""Environment configuration with Docker/Kubernetes file-secret support.

``env(name)`` reads a variable from the process environment, but if ``<name>_FILE`` is
set it reads the value from that file instead: the widely supported convention for
injecting secrets from Docker secrets or a Kubernetes secret mount without placing them
in the environment. OWASP is direct about why that matters: environment variables are
"generally accessible to all processes and may be included in logs or system dumps", and
so are the least preferred way to pass a secret. Applied to the sensitive settings
(``APP_SECRET_KEY``, ``SMTP_PASSWORD``, ``DB_URI``, ``LLM_API_KEY``) so an operator can
mount them as files.

Both failure modes are fatal, matching the ``file_env`` helper the Docker official
images (postgres, mysql) define for this same convention:

* ``<name>`` and ``<name>_FILE`` both set is a configuration error, not a precedence
  question. Whichever were chosen, an operator has expressed two intentions and at most
  one of them is being honoured.
* ``<name>_FILE`` naming a file that cannot be read is a configuration error too.
  Falling back to the plain variable (or worse, to a default) is how a missing
  secret mount becomes an authentication failure far from its cause, and for
  ``APP_SECRET_KEY`` it would mean silently encrypting the vault under a different key.

Neither is recoverable at runtime, so both raise rather than warn. A container that will
not start names the problem; one that starts with the wrong secret does not.
"""

from __future__ import annotations

import os
from pathlib import Path

from elbi_core.errors import ConfigError


def env(name: str, default: str | None = None) -> str | None:
    """The value of ``name``, or of the file named by ``<name>_FILE``, else ``default``.

    Raises:
        ConfigError: if both ``name`` and ``<name>_FILE`` are set, or if ``<name>_FILE``
            is set and cannot be read.
    """
    file_path = os.environ.get(f"{name}_FILE")
    if not file_path:
        return os.environ.get(name, default)

    if os.environ.get(name) is not None:
        raise ConfigError(
            f"{name} and {name}_FILE are both set, but they are exclusive. "
            f"Supply the value either directly or as a file, not both."
        )
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(
            f"{name}_FILE is set to {file_path!r} but it could not be read: {exc}. "
            f"Refusing to fall back, because a secret that was meant to be loaded and "
            f"was not is a configuration error rather than a missing value."
        ) from exc
