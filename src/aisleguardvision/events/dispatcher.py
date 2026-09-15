"""Alert dispatch.

Delivery runs on its own thread, behind a bounded queue. This is not incidental:
a webhook to a store's ticketing system can take seconds or hang entirely, and
doing that on the inference thread would stall video processing for every camera
that thread serves. **No network call ever happens on the analysis path.**

Providers are pluggable. Shipped: console and webhook. The obvious next ones --
SMS, email, Slack, mobile push, a dashboard websocket -- are each a subclass of
:class:`AlertProvider` with no changes anywhere else.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod

from ..core.config import AlertsConfig, WebhookConfig
from ..core.logging import get_logger
from ..core.metrics import MetricNames, MetricsRegistry, get_metrics
from ..core.types import SecurityEvent, ThreatLevel
from ..utils.sanitize import sanitize_url
from .models import SecurityEventModel

logger = get_logger(__name__)


class AlertProvider(ABC):
    """A delivery channel for security events."""

    #: Shown in logs and metrics.
    name: str = "provider"

    @abstractmethod
    def send(self, event: SecurityEventModel) -> bool:
        """Deliver one event. Returns success. Must not raise."""

    def close(self) -> None:  # noqa: B027 - optional hook, not an abstract method
        """Release resources held by the provider.

        Deliberately concrete and empty: most providers hold nothing to
        release, and forcing every subclass to implement a no-op would be
        noise.
        """


class ConsoleAlertProvider(AlertProvider):
    """Writes the alert, with its full reasoning, to the log.

    Always enabled by default: an operator running the MVP on a laptop needs to
    see *why* an alert fired, and the evidence breakdown is the product.
    """

    name = "console"

    def send(self, event: SecurityEventModel) -> bool:
        lines = [
            "",
            "=" * 78,
            f"  ALERT  {event.threat_level.value}  risk={event.risk_score:.1f}",
            f"  camera={event.camera_id}  track_id={event.person_id}  "
            f"state={event.behavior_state.value}",
            f"  event_id={event.event_id}",
            f"  time={event.timestamp_iso}",
            "",
            "  Possible concealment behavior detected - human review recommended.",
            "  This is NOT a determination that a theft occurred.",
            "",
            "  Supporting observations:",
        ]
        if event.positive_evidence:
            lines.extend(f"    + {text}" for text in event.positive_evidence)
        else:
            lines.append("    (none)")
        lines.append("")
        lines.append("  Observations against:")
        if event.negative_evidence:
            lines.extend(f"    - {text}" for text in event.negative_evidence)
        else:
            lines.append("    (none)")
        if event.clip_path:
            lines.extend(["", f"  clip:     {event.clip_path}"])
        if event.snapshot_path:
            lines.append(f"  snapshot: {event.snapshot_path}")
        lines.append("=" * 78)
        logger.warning("\n".join(lines))
        return True


class WebhookAlertProvider(AlertProvider):
    """POSTs the event payload to an HTTP endpoint.

    Retries with exponential backoff. The bearer token comes from the
    environment and is never logged; the URL is sanitized before it appears in
    any log record.
    """

    name = "webhook"

    def __init__(self, config: WebhookConfig) -> None:
        self.config = config
        self.safe_url = sanitize_url(config.url)
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests

            session = requests.Session()
            session.headers.update(
                {
                    "Content-Type": "application/json",
                    "User-Agent": "AisleGuardVision/0.1",
                }
            )
            if self.config.token:
                session.headers["Authorization"] = f"Bearer {self.config.token}"
            self._session = session
        return self._session

    def send(self, event: SecurityEventModel) -> bool:
        payload = event.to_webhook_payload()
        delay = self.config.backoff_base_seconds

        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self._get_session().post(
                    self.config.url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_tls,
                )
                if 200 <= response.status_code < 300:
                    logger.info(
                        "webhook delivered",
                        extra={
                            "fields": {
                                "event_id": event.event_id,
                                "url": self.safe_url,
                                "status": response.status_code,
                                "attempt": attempt,
                            }
                        },
                    )
                    return True

                # 4xx other than 408/429 will not succeed on retry.
                if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                    logger.error(
                        "webhook rejected the event; not retrying",
                        extra={
                            "fields": {
                                "event_id": event.event_id,
                                "url": self.safe_url,
                                "status": response.status_code,
                            }
                        },
                    )
                    return False

                reason = f"HTTP {response.status_code}"
            except Exception as exc:
                reason = type(exc).__name__

            if attempt > self.config.max_retries:
                logger.error(
                    "webhook delivery failed after all retries",
                    extra={
                        "fields": {
                            "event_id": event.event_id,
                            "url": self.safe_url,
                            "attempts": attempt,
                            "reason": reason,
                        }
                    },
                )
                return False

            logger.warning(
                "webhook delivery failed; retrying",
                extra={
                    "fields": {
                        "event_id": event.event_id,
                        "url": self.safe_url,
                        "attempt": attempt,
                        "reason": reason,
                        "retry_in_s": round(delay, 2),
                    }
                },
            )
            time.sleep(delay)
            delay = min(delay * 2.0, self.config.backoff_max_seconds)
        return False

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


class CallbackAlertProvider(AlertProvider):
    """Delivers to an in-process callable. Used by the API and by tests."""

    name = "callback"

    def __init__(self, callback, name: str = "callback") -> None:
        self._callback = callback
        self.name = name

    def send(self, event: SecurityEventModel) -> bool:
        try:
            self._callback(event)
        except Exception:
            logger.exception("alert callback raised", extra={"fields": {"provider": self.name}})
            return False
        return True


class AlertDispatcher:
    """Queues events and delivers them on a background thread."""

    def __init__(
        self,
        config: AlertsConfig,
        providers: list[AlertProvider] | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics or get_metrics()
        self.providers: list[AlertProvider] = providers if providers is not None else []

        if providers is None:
            if config.console:
                self.providers.append(ConsoleAlertProvider())
            if config.webhook.enabled and config.webhook.url:
                self.providers.append(WebhookAlertProvider(config.webhook))

        self._queue: queue.Queue[SecurityEventModel | None] = queue.Queue(maxsize=config.queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.dispatched = 0
        self.dropped = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="alert-dispatcher", daemon=True
        )
        self._thread.start()
        logger.info(
            "alert dispatcher started",
            extra={
                "fields": {
                    "providers": ",".join(p.name for p in self.providers) or "none",
                    "queue_size": self.config.queue_size,
                }
            },
        )

    def stop(self, timeout: float = 15.0) -> None:
        """Stop after draining the queue, so a final alert is not lost."""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._stop.set()
        for provider in self.providers:
            try:
                provider.close()
            except Exception:  # pragma: no cover
                logger.exception("error closing alert provider")

    def add_provider(self, provider: AlertProvider) -> None:
        self.providers.append(provider)

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, event: SecurityEvent) -> bool:
        """Enqueue an event for delivery. Returns immediately.

        Called from the analysis thread; this method must never block and never
        perform I/O.
        """
        model = SecurityEventModel.from_event(event)
        try:
            self._queue.put_nowait(model)
        except queue.Full:
            self.dropped += 1
            logger.error(
                "alert queue is full; dropping the alert rather than stalling inference",
                extra={
                    "fields": {
                        "event_id": event.event_id,
                        "camera_id": event.camera_id,
                        "dropped_total": self.dropped,
                    }
                },
            )
            return False

        self.metrics.inc(MetricNames.ALERTS_GENERATED, camera_id=event.camera_id)
        return True

    def dispatch_sync(self, event: SecurityEvent) -> bool:
        """Deliver immediately on the calling thread.

        For the CLI and tests only. Never call this from the analysis path.
        """
        return self._deliver(SecurityEventModel.from_event(event))

    def _dispatch_loop(self) -> None:
        while True:
            try:
                model = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if model is None:
                return
            try:
                self._deliver(model)
            finally:
                self._queue.task_done()

    def _deliver(self, model: SecurityEventModel) -> bool:
        delivered_any = False
        for provider in self.providers:
            try:
                ok = provider.send(model)
            except Exception:
                logger.exception(
                    "alert provider raised", extra={"fields": {"provider": provider.name}}
                )
                ok = False
            if ok:
                delivered_any = True
                if provider.name == "webhook":
                    self.metrics.inc(MetricNames.WEBHOOK_DELIVERIES)
            elif provider.name == "webhook":
                self.metrics.inc(MetricNames.WEBHOOK_FAILURES)
        if delivered_any:
            self.dispatched += 1
        return delivered_any

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()


#: Module-level dispatcher used by the trigger_alert() convenience function.
_default_dispatcher: AlertDispatcher | None = None


def set_default_dispatcher(dispatcher: AlertDispatcher | None) -> None:
    """Register the dispatcher that :func:`trigger_alert` routes through."""
    global _default_dispatcher
    _default_dispatcher = dispatcher


def trigger_alert(
    person_id: int,
    crop_img=None,
    *,
    camera_id: str = "unknown",
    risk_score: float = 100.0,
    reason: str = "manual alert",
) -> SecurityEvent | None:
    """Convenience entry point for raising an alert by hand.

    Provided because it is a natural thing to reach for, but it is *not* a
    standalone shortcut around the architecture: it builds a real
    :class:`SecurityEvent`, routes it through the configured
    :class:`AlertDispatcher`, and is subject to the same providers, the same
    payload schema and the same audit record as an alert raised by the behavior
    engine.

    An alert raised this way is explicitly marked as manually triggered, so a
    reviewer is never shown a hand-made alert dressed up as an inference.

    Returns ``None`` when no dispatcher has been configured, rather than
    silently creating global state.
    """
    from ..core.types import BehaviorState, EventType, EvidenceType, RiskContribution

    if _default_dispatcher is None:
        logger.error(
            "trigger_alert called with no dispatcher configured; "
            "call set_default_dispatcher() during startup"
        )
        return None

    event = SecurityEvent(
        event_id=SecurityEvent.new_id(),
        camera_id=camera_id,
        person_id=person_id,
        timestamp=time.time(),
        risk_score=risk_score,
        threat_level=ThreatLevel.from_score(risk_score),
        behavior_state=BehaviorState.REVIEW_ALERT,
        event_type=EventType.POSSIBLE_CONCEALMENT,
        positive_evidence=[f"Manually triggered: {reason}"],
        negative_evidence=[],
        contributions=[
            RiskContribution(
                evidence_type=EvidenceType.SHELF_INTERACTION,
                description=f"Manually triggered: {reason}",
                base_weight=0.0,
                confidence=1.0,
                applied_weight=0.0,
            )
        ],
        track_metadata={"manual": True, "has_crop": crop_img is not None},
    )
    _default_dispatcher.dispatch(event)
    return event
