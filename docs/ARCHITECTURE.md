# AisleGuard Vision — Architecture

## What this system is, and what it is not

AisleGuard Vision analyzes live security-camera streams and identifies
**behavior sequences** consistent with possible merchandise concealment, so a
human can review them.

An alert means exactly one thing:

> **Possible concealment behavior detected — human review recommended.**

It is **not** a determination that a theft occurred, and must never be
presented or acted on as one. The system performs **no** facial recognition,
identity recognition, or race, ethnicity, gender, age or other demographic
classification of any kind. Person identifiers are temporary computer-vision
track ids, scoped to one camera session and reused once a track ends.

---

## Pipeline

```
                 ┌────────────────┐
  IP cameras ───▶│ Camera decode  │  one thread per camera
  webcam         │    worker      │  RTSP reconnect, stall detection
  MP4 file       └────────┬───────┘
                          │  Frame
                          ▼
                 ┌────────────────┐
                 │  Frame queue   │  BOUNDED, drop-oldest
                 └────────┬───────┘
                          │
              ┌───────────┴──────────┐
              ▼                      ▼
      ┌───────────────┐      ┌───────────────┐
      │ Circular      │      │ Frame         │  decides per frame:
      │ frame buffer  │      │ scheduler     │  detect? pose? on whom?
      │ (pre-event)   │      └───────┬───────┘
      └───────┬───────┘              │
              │                      ▼
              │              ┌───────────────┐
              │              │ Detector      │  YOLO11 via DetectorBackend
              │              │ backend       │  list-in / list-out (batchable)
              │              └───────┬───────┘
              │                      │  Detection[]
              │         ┌────────────┼────────────┐
              │         ▼            ▼            ▼
              │   ┌──────────┐ ┌──────────┐ ┌──────────┐
              │   │  people  │ │ objects  │ │containers│
              │   └────┬─────┘ └────┬─────┘ └────┬─────┘
              │        ▼            ▼            │
              │  ┌──────────┐  ┌──────────┐      │
              │  │ ByteTrack│  │ Product  │      │
              │  │  person  │  │ detector │      │
              │  │  tracker │  └────┬─────┘      │
              │  └────┬─────┘       ▼            │
              │       │       ┌──────────┐       │
              │       │       │   Item   │       │
              │       │       │ tracker  │       │
              │       │       │ (ladder) │       │
              │       │       └────┬─────┘       │
              │       ▼            │             │
              │  ┌──────────┐      │             │
              │  │Conditional│     │             │
              │  │   pose    │     │             │
              │  └────┬──────┘     │             │
              │       ▼            │             │
              │  ┌──────────────┐  │             │
              │  │ pose/person  │  │             │
              │  │ association  │  │             │
              │  │ (geometric)  │  │             │
              │  └────┬─────────┘  │             │
              │       └─────┬──────┘             │
              │             ▼                    │
              │      ┌──────────────┐            │
              │      │  hand/item   │            │
              │      │ association  │            │
              │      └──────┬───────┘            │
              │             │                    │
              │             ▼                    │
              │   ┌─────────────────────┐        │
              │   │  Temporal behavior  │◀───────┘
              │   │       engine        │◀─── zones (shelf/basket/cart/…)
              │   │  (evidence ledger)  │
              │   └──────────┬──────────┘
              │              ▼
              │      ┌───────────────┐
              │      │  Risk engine  │  clamped sum of weighted evidence
              │      └───────┬───────┘
              │              ▼
              │      ┌───────────────┐
              │      │   Cooldown    │  per person, escalation override
              │      └───────┬───────┘
              │              ▼
              └─────▶┌───────────────┐
                     │   Incident    │  clip + snapshot + JSON
                     │   recorder    │  background writer thread
                     └───────┬───────┘
                             ▼
                     ┌───────────────┐
                     │     Alert     │  console, webhook (off the hot path)
                     │  dispatcher   │
                     └───────┬───────┘
                             ▼
                     ┌───────────────┐
                     │  FastAPI      │  /health /metrics /cameras /events
                     └───────────────┘
```

---

## The seven decisions that shape everything else

### 1. A hard dependency firewall around the models

`behavior/`, `tracking/`, `events/` and `core/` import **no torch, no
ultralytics, and no cv2 on any decision path**. Torch and Ultralytics are
reachable only through `inference/ultralytics_backend.py`, which is imported
lazily.

Consequences, all of them deliberate:

- the full test suite (383 tests) runs on a machine with no GPU and no weights;
- the behavior simulator runs anywhere, so risk tuning does not need hardware;
- swapping in TensorRT, ONNX, Triton or DeepStream cannot change behavior
  logic, because that logic cannot see the backend.

You can verify the firewall holds:

```bash
python - <<'PY'
import sys
import aisleguardvision.behavior.engine, aisleguardvision.tracking.person_tracker
assert "torch" not in sys.modules and "ultralytics" not in sys.modules
print("firewall intact")
PY
```

### 2. Evidence-ledger risk, never a black-box score

The risk engine does not run a model. It sums configured weights over the
discrete, human-readable observations the behavior engine raised:

```
raw   = Σ weight(evidence) × confidence(evidence) × zone_multiplier
score = clamp(raw, 0, 100)
```

A learned "shoplifting probability" would be untunable per store, unauditable
after the fact, and impossible to explain to the person being reviewed. Every
point in an AisleGuard score is attributable to a named observation, and the
full arithmetic ships with the alert.

### 3. Episodes, not running totals

Evidence is scoped to an **interaction episode** — one shelf approach through
to its resolution. A benign explanation (basket, cart, shelf return, phone)
doesn't merely subtract points: it *terminates the episode and clears its
positive evidence*. This is the single largest false-positive lever in the
design. See [FALSE_POSITIVES.md](FALSE_POSITIVES.md).

### 4. Seconds, never frames

Store cameras run at 10, 15, 20, 25 and 30 FPS, and any one camera's effective
rate moves with network load and scheduler pressure. Every threshold in the
system is a duration. The Kalman filter takes `dt` as a parameter on every
predict step; track lifetimes, occlusion ladders, dwell requirements and
cooldowns are all in seconds.

A test asserts this directly: the same occlusion is carried identically at 10
and 30 FPS (`test_track_lifetimes_are_time_based_not_frame_based`).

### 5. Batch-shaped backend API from day one

```python
def infer(self, images: list[np.ndarray]) -> list[list[Detection]]: ...
```

The single-camera MVP passes a list of one. A multi-camera GPU worker passes
eight frames from eight cameras and gets one GPU invocation. No signature
change, no tracker change, no behavior change. See [SCALING.md](SCALING.md).

### 6. Bounded queues, drop-oldest

For live video a fresh frame is worth more than a backlog. An unbounded queue
does not prevent overload — it converts an overload into unbounded latency and
then an OOM. `FrameQueue` is bounded and evicts the oldest frame, recording the
drop on the frame that *is* delivered so the pipeline knows what it missed.

### 7. Nothing blocks the inference path

Clip encoding runs on the incident recorder's writer thread. Webhook delivery
runs on the dispatcher's thread. A webhook to a store's ticketing system can
hang for seconds; doing that inline would stall every camera the thread serves.
When either queue fills, work is dropped with a logged error rather than
back-pressuring video.

---

## Module map

| Module | Responsibility | Imports models? |
|---|---|---|
| `core/types.py` | Every domain type crossing a stage boundary | no |
| `core/config.py` | YAML + `${ENV}` → validated pydantic models | no |
| `core/logging.py` | Structured logging, credential sanitization | no |
| `core/metrics.py` | Counters, gauges, rate meters, histograms, Prometheus | no |
| `camera/stream.py` | Source abstraction, reconnect, stall detection | cv2 only |
| `camera/worker.py` | One decode thread per camera | cv2 only |
| `camera/frame_buffer.py` | Circular pre-event buffer, bounded queue | cv2 only |
| `camera/manager.py` | Camera lifecycle, per-camera failure isolation | cv2 only |
| `inference/backend.py` | `DetectorBackend` ABC, registry, error isolation | no |
| `inference/ultralytics_backend.py` | **The only torch/ultralytics importer** | yes |
| `inference/device.py` | CUDA → MPS → CPU selection | lazily |
| `inference/detector.py` | Partition detections into people/objects/containers | no |
| `inference/pose.py` | Crop-mode conditional pose, coordinate mapping | no |
| `inference/product_detector.py` | Merchandise abstraction + honest fallback | lazily |
| `inference/scheduler.py` | Adaptive detection/pose rate governor | no |
| `tracking/kalman.py` | Constant-velocity filter, IoU, greedy matching | no |
| `tracking/person_tracker.py` | ByteTrack, in-repo, second-based lifetimes | no |
| `tracking/item_tracker.py` | Merchandise tracks + occlusion ladder | no |
| `tracking/association.py` | Pose↔person, item↔wrist, wrist history | no |
| `behavior/geometry.py` | Polygons, storage regions, motion profiles | no |
| `behavior/zones.py` | Zone registry and queries | no |
| `behavior/evidence.py` | Evidence ledger and factories | no |
| `behavior/state_machine.py` | Sequence ordering and benign branching | no |
| `behavior/engine.py` | The temporal reasoning | no |
| `behavior/risk.py` | Explainable scoring | no |
| `events/` | Cooldown, incident recording, alert dispatch | cv2 only |
| `visualization/overlay.py` | OpenCV drawing (never influences a decision) | cv2 only |
| `api/` | FastAPI read-mostly view of a running pipeline | no |
| `simulation/` | Scenarios + harness over the real stack | no |

---

## The merchandise-detection limitation

**A COCO-trained YOLO model has no generic retail-product class.** It cannot
recognize a wig, a hair bundle, a jar of gel, a cosmetics package, a hair
accessory, or most other store inventory. No threshold or post-processing
changes that.

The system handles this by stating it, not by pretending:

1. `ProductDetector` exists as the interface the product needs.
2. `ZoneOnlyProductDetector` ships as the honest default — it detects nothing
   and reports `provides_merchandise_detection = False`.
3. `CocoProxyProductDetector` surfaces the few carryable COCO classes that
   genuinely *are* merchandise in some categories (bottle, cup, book), and says
   plainly that this covers a narrow slice of inventory.
4. `CustomModelProductDetector` is a working implementation: point
   `models.product` at YOLO-format weights trained on store inventory and every
   class maps to `MERCHANDISE`, keeping the SKU label for the incident record.

**In zone-only mode the risk score is capped below the alert threshold**
(`behavior.zone_only_risk_ceiling`, default 59 against a threshold of 85). Shelf
geometry plus wrist kinematics is real evidence, but it is not sufficient
grounds to send a staff member to confront a shopper. The config layer refuses
to start if that invariant is violated, and a test asserts it
(`test_zone_only_mode_cannot_reach_the_alert_threshold`).

### Plugging in a trained retail model

```yaml
# config/detection.yaml
models:
  product: models/retail_v1.pt
```

Nothing else changes. The item tracker, hand/item associator, behavior engine
and risk weights already expect item-level evidence — they simply start
receiving it, and the risk ceiling lifts automatically.

---

## Data flow of one alert

1. Wrist enters a shelf zone and stays ≥ `min_shelf_interaction_seconds`
   → `SHELF_INTERACTION` (+10)
2. A merchandise track links to that wrist for ≥ `min_item_association_seconds`,
   scored across distance / IoU / motion-correlation / persistence
   → `STABLE_HAND_ITEM_ASSOCIATION` (+15)
3. The item leaves its origin zone while held
   → `ITEM_REMOVED_FROM_SHELF` (+10)
4. The wrist travels toward a pose-derived storage region **and dwells there**
   → `HAND_MOVED_TO_STORAGE_REGION` (+15)
5. With a downward/inward motion profile over a 1 s window
   → `CONCEALMENT_MOTION_PROFILE` (+10)
6. The item stops being detected near that region
   → `ITEM_DISAPPEARED_NEAR_STORAGE` (+25)
7. It stays unobserved past `item_missing_timeout_seconds`
   → `ITEM_REMAINS_MISSING` (+20)
8. No basket, cart or shelf-return was observed
   → recorded at weight 0, for the reviewer

Score ≥ 85 **and** the required evidence present → `SecurityEvent` → clip +
snapshot + JSON → console/webhook → 30 s per-person cooldown.

`RiskEngine.should_alert()` additionally refuses to alert if the score somehow
crosses the threshold without `STABLE_HAND_ITEM_ASSOCIATION` and
`ITEM_DISAPPEARED_NEAR_STORAGE` present — that combination would be a scoring
misconfiguration, not an incident.

---

## Threading model

| Thread | Owns | Count |
|---|---|---|
| decode worker | one camera's `VideoCapture` | one per camera |
| main / analysis | detection, tracking, behavior, risk | one per process |
| incident writer | clip encoding, snapshot writing | one |
| alert dispatcher | webhook and console delivery | one |
| uvicorn | API requests | pool, optional |

The analysis loop is single-threaded across cameras in the MVP. At one to a
handful of cameras the GPU is the bottleneck, not the loop, and a single
threaded loop keeps the control flow inspectable. Fan-out is by **process** —
see [SCALING.md](SCALING.md) — and each worker process runs this same loop over
its own camera subset.

---

## Extension points

| To add | Implement | Nothing else changes |
|---|---|---|
| A new inference runtime | `DetectorBackend` subclass + `register_backend` | behavior, tracking, risk |
| Merchandise detection | `models.product` config, or a `ProductDetector` | the whole pipeline |
| A new alert channel (SMS, Slack, push) | `AlertProvider` subclass | dispatcher, cooldown, payloads |
| A new evidence type | `EvidenceType` member + factory + weight | scoring is automatic |
| A new zone kind | `ZoneKind` member | registry queries are generic |
| Incident storage in a database | `IncidentStore` replacement | recorder, API |
