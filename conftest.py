"""Root pytest configuration shared by every test path.

Registers Hypothesis profiles so property-based tests behave predictably. The
``ci`` profile is deterministic (a fixed example stream) and prints a reproduction
blob on failure, so a failing property test can be reproduced exactly rather than
vanishing on the next run. Deadlines are disabled to avoid timing-based flakes.
Set ``HYPOTHESIS_PROFILE`` to override; ``ci`` is selected automatically under CI.
"""

from __future__ import annotations

import os

from hypothesis import settings

settings.register_profile("dev", deadline=None)
settings.register_profile("ci", deadline=None, derandomize=True, print_blob=True)

_default_profile = "ci" if os.environ.get("CI") else "dev"
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", _default_profile))
