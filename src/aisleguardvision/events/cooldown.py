"""Per-person alert cooldown.

Without this, a single sustained concealment sequence generates an alert on
every analysis frame -- hundreds of notifications for one event, which is the
fastest way to make staff ignore the system entirely.

A second alert for the same person is allowed only when one of these holds:

1. the cooldown has expired;
2. risk has meaningfully *escalated* (configurable delta), meaning new
   evidence arrived rather than the same evidence being re-scored;
3. a genuinely distinct event type occurred.

Rule 2 is the important one: suppressing a REVIEW-level alert that has since
become a HIGH-RISK one would hide exactly the escalation staff need.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..core.logging import get_logger
from ..core.types import EventType

logger = get_logger(__name__)


@dataclass(slots=True)
class CooldownRecord:
    """Last alert issued for one tracked person."""

    timestamp: float
    risk_score: float
    event_type: EventType
    event_id: str


@dataclass(slots=True)
class CooldownDecision:
    """Whether an alert may be issued, and why."""

    allowed: bool
    reason: str
    seconds_remaining: float = 0.0


class AlertCooldown:
    """Cooldown state for all tracked people on all cameras."""

    def __init__(self, cooldown_seconds: float, escalation_delta: float = 10.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.escalation_delta = escalation_delta
        self._records: dict[tuple[str, int], CooldownRecord] = {}
        self._lock = threading.Lock()
        self.suppressed_count = 0

    def check(
        self,
        camera_id: str,
        person_id: int,
        timestamp: float,
        risk_score: float,
        event_type: EventType = EventType.POSSIBLE_CONCEALMENT,
    ) -> CooldownDecision:
        """Decide whether an alert may fire. Does not record anything."""
        key = (camera_id, person_id)
        with self._lock:
            record = self._records.get(key)

        if record is None:
            return CooldownDecision(True, "first alert for this track")

        elapsed = timestamp - record.timestamp
        if elapsed >= self.cooldown_seconds:
            return CooldownDecision(True, f"cooldown expired after {elapsed:.1f}s")

        if event_type is not record.event_type:
            return CooldownDecision(
                True, f"distinct event type ({event_type.value} vs {record.event_type.value})"
            )

        escalation = risk_score - record.risk_score
        if escalation >= self.escalation_delta:
            return CooldownDecision(
                True,
                f"risk escalated by {escalation:.1f} ({record.risk_score:.1f} -> {risk_score:.1f})",
            )

        return CooldownDecision(
            False,
            f"within {self.cooldown_seconds:.0f}s cooldown and risk has not escalated",
            seconds_remaining=self.cooldown_seconds - elapsed,
        )

    def record(
        self,
        camera_id: str,
        person_id: int,
        timestamp: float,
        risk_score: float,
        event_id: str,
        event_type: EventType = EventType.POSSIBLE_CONCEALMENT,
    ) -> None:
        """Register that an alert was issued."""
        with self._lock:
            self._records[(camera_id, person_id)] = CooldownRecord(
                timestamp=timestamp,
                risk_score=risk_score,
                event_type=event_type,
                event_id=event_id,
            )

    def note_suppressed(self) -> None:
        self.suppressed_count += 1

    def forget(self, camera_id: str, person_id: int) -> None:
        """Drop state for a track that no longer exists.

        Track ids are reused after a track is removed, so stale cooldown state
        would otherwise suppress a genuine alert for a *different* shopper who
        happened to inherit the id.
        """
        with self._lock:
            self._records.pop((camera_id, person_id), None)

    def retain_only(self, camera_id: str, live_person_ids: set[int]) -> None:
        """Forget every person on this camera that is no longer tracked."""
        with self._lock:
            for key in list(self._records):
                if key[0] == camera_id and key[1] not in live_person_ids:
                    del self._records[key]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    @property
    def tracked_count(self) -> int:
        with self._lock:
            return len(self._records)
