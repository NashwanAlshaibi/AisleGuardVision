"""AisleGuard Vision - AI-assisted retail loss-prevention behavioral analysis.

AisleGuard Vision analyzes live security-camera streams and identifies
*behavior sequences* consistent with possible merchandise concealment, so that
store personnel can review them.

It does not decide that anyone committed theft. An alert means exactly one
thing:

    Possible concealment behavior detected - human review recommended.

The system performs no facial recognition, identity recognition, or race,
ethnicity, gender, age or other demographic classification of any kind.
Tracking identifiers are temporary computer-vision track ids, scoped to a
single camera session and reused after the track ends.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
