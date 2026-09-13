# AisleGuard Vision

**Real-time retail loss-prevention behavioral analysis.** AisleGuard Vision
analyzes live security-camera streams and identifies *behavior sequences*
consistent with possible merchandise concealment, so that store personnel can
review them.

---

> ## What an alert means
>
> **Possible concealment behavior detected — human review recommended.**
>
> An alert is **not** a determination that a theft occurred, and must never be
> presented or acted on as one. It is a prompt for a person to watch a ten
> second clip and decide.
>
> AisleGuard Vision performs **no** facial recognition, identity recognition,
> or race, ethnicity, gender, age or other demographic classification of any
> kind. Person identifiers are temporary computer-vision track ids, scoped to a
> single camera session and reused once a track ends. These are not
> configuration options — they are absent from the architecture.

---

## Why this is not just an object detector

Naive approaches fire on a single suspicious-looking frame and are unusable
within a day. AisleGuard reasons **over time** and requires a sequence of
mutually supporting observations:

```
  1. wrist interacts with a shelf zone, for a minimum dwell
  2. a merchandise item becomes stably associated with that wrist
  3. the item leaves the shelf zone while held
  4. that wrist travels toward a pose-derived storage region, and stays there
  5. with a downward/inward motion profile
  6. the item moves WITH the wrist
  7. the item becomes occluded near that region
  8. and stays unobserved past the occlusion timeout
  9. no basket or cart placement was observed
 10. no return to a shelf was observed
```

All of these are explicitly **normal** and cannot escalate risk on their own:
picking up merchandise, browsing, putting it back, holding it, checking a
phone, reaching into a pocket, adjusting clothing, carrying a bag, loading a
basket, or being briefly occluded.

**Measured behavior of the shipped configuration** (`python
scripts/simulate_behavior.py`):

| Scenario | Peak risk | Outcome |
|---|---|---|
| Phone taken from pocket, used, pocketed | **0.0** | no alert |
| Reaching into a pocket, no merchandise | **0.0** | no alert |
| Browsing a shelf | 10.0 | no alert |
| Item into a basket / cart | 22.0 | no alert |
| Item picked up and returned | 27.5 | no alert |
| Same movement, no merchandise detector | 32.5 | no alert *(capped by design)* |
| Held item briefly occluded | 35.0 | no alert |
| **Full concealment sequence** | **100.0** | **alert** |

The gap between 35.0 and the threshold of 85 is the safety margin. See
[docs/FALSE_POSITIVES.md](docs/FALSE_POSITIVES.md).

---

## Architecture

```
IP cameras / webcam / MP4
        │
        ▼
┌──────────────────┐   one thread per camera; RTSP reconnect, stall detection
│  Camera decode   │
└────────┬─────────┘
         ▼
┌──────────────────┐   BOUNDED, drop-oldest: a fresh frame beats a backlog
│   Frame queue    │
└────────┬─────────┘
         ▼
┌──────────────────┐   detect? pose? on whom? — three independent rate budgets
│ Frame scheduler  │
└────────┬─────────┘
         ▼
┌──────────────────┐   YOLO11 behind DetectorBackend; list-in/list-out
│ Person detection │   → ByteTrack (in-repo, second-based lifetimes)
└────────┬─────────┘
         ▼
┌──────────────────┐   crop-mode, conditional; matched to tracks GEOMETRICALLY
│ Pose estimation  │
└────────┬─────────┘
         ▼
┌──────────────────┐   zones + item tracking + hand/item association
│ Retail context   │
└────────┬─────────┘
         ▼
┌──────────────────┐   per-person evidence ledger, episodes, benign branching
│ Behavior engine  │
└────────┬─────────┘
         ▼
┌──────────────────┐   clamped sum of weighted, human-readable evidence
│   Risk engine    │   NOT a black-box "shoplifting probability"
└────────┬─────────┘
         ▼
┌──────────────────┐   clip + snapshot + JSON → console / webhook → API
│ Incident + alert │
└──────────────────┘
```

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Setup

Requires **Python 3.11+**.

```bash
git clone <repo>
cd aisleguardvision

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That is enough to run the behavior engine, the simulator, the API and the full
test suite. **No GPU and no model weights required** — the behavior logic
imports no deep-learning framework at all, which is enforced by a test.

### Enabling live detection

```bash
pip install -r requirements-inference.txt
```

For CUDA, install torch from NVIDIA's index **first**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-inference.txt
```

On Jetson, do **not** install torch from PyPI — see [docs/JETSON.md](docs/JETSON.md).

For a GUI window (`--display`), you need the non-headless OpenCV build:

```bash
pip install -r requirements-display.txt
```

YOLO11 weights (`yolo11n.pt`, `yolo11n-pose.pt`) download automatically on
first run, about 11 MB total.

### GPU detection

Acceleration is selected automatically in the order **CUDA → MPS → CPU**.
Nothing assumes CUDA exists. Check what was chosen:

```bash
python -c "from aisleguardvision.inference.device import select_device, log_device; \
           log_device(select_device('auto'))"
```

Override with `--device cuda:1` / `--device cpu`, or `models.device` in config.
FP16 is used automatically on CUDA devices with compute capability ≥ 6.0.

---

## Running

```bash
# Webcam
python -m aisleguardvision.main --source 0 --display

# Video file
python -m aisleguardvision.main --source ./data/samples/store.mp4 --display

# RTSP camera
python -m aisleguardvision.main --source "rtsp://user:pass@host:554/stream" --display

# Headless (edge node / server)
python -m aisleguardvision.main --headless

# Every enabled camera in config/cameras.yaml, plus the API
python -m aisleguardvision.main --headless --api
```

No sample footage ships with the repository — real store footage of real
shoppers is not something to commit. Generate a synthetic clip to check the
plumbing end to end:

```bash
python scripts/make_sample_video.py
python -m aisleguardvision.main --source ./data/samples/store.mp4 --display
```

Those figures are schematic, so a real detector finds little in them. Use them
to verify decode, buffering, overlay, recording and shutdown; use real footage
to verify detection quality.

### Useful flags

| Flag | Effect |
|---|---|
| `--display` / `--headless` | show a window / never open one |
| `--camera cam_001` | run only this configured camera (repeatable) |
| `--device cuda:0` | override device selection |
| `--detection-fps 5` | override the detection rate |
| `--alert-threshold 90` | override the alert threshold |
| `--product-model path.pt` | supply a custom retail merchandise model |
| `--no-pose` | detection only (much faster, no behavior reasoning) |
| `--no-record` | disable clip and snapshot recording |
| `--record-output out.mp4` | write the annotated video |
| `--api --api-port 8000` | start the API alongside the pipeline |
| `--log-format json` | single-line JSON logs for aggregation |
| `--max-seconds 60` | exit after a fixed time (useful for CI) |
| `--loop` | loop a file source |

### The display

```
┌─────────────────────────────┐
│ AisleGuard Vision           │      ID: 14
│ CAMERA: cam_001             │      STATE: ITEM_ASSOCIATED
│                             │      RISK: 42%
│ VIDEO:      29.8 FPS        │
│ DETECTION:  10.1 FPS        │   ┌──────────────────────────────┐
│ POSE:        7.8 FPS        │   │ TRACK 14  ELEVATED  42       │
│ INFERENCE:    31 ms         │   │   +10 Shelf interaction …    │
│ TRACKS:        5            │   │   +15 Right wrist associat…  │
│ QUEUE:         1            │   │   -30 Item remains visible…  │
│ DROPPED:      23            │   └──────────────────────────────┘
└─────────────────────────────┘
```

Person boxes, track ids, pose skeletons, wrists, zone polygons, item tracks,
estimated storage regions, behavior state, risk, threat level, FPS and latency.
The evidence panel shows the reasoning for the highest-risk person in frame —
if an operator cannot see *why* a score is what it is, they cannot trust or
tune it.

Press `q` or `Esc` to quit.

---

## Configuration

Three YAML files. **No behavioral constant is hard-coded in the source.**

| File | Contains |
|---|---|
| `config/app.yaml` | behavior thresholds, risk weights, recording, storage, alerting, API, logging |
| `config/detection.yaml` | models, inference, frame scheduling, tracking |
| `config/cameras.yaml` | cameras and their zones |

Values of the form `${VAR}` or `${VAR:-default}` are resolved from the
environment at load time:

```yaml
cameras:
  - id: cam_001
    name: Test Camera
    source: ${CAM_001_RTSP}      # never a literal credential
    enabled: true
```

```bash
cp .env.example .env             # .env is git-ignored
export CAM_001_RTSP="rtsp://viewer:realpassword@10.0.0.5:554/Streaming/Channels/101"
```

A camera whose variable is unset is **skipped with a warning**, not a crash, so
a dev box with one camera's credentials can load a config listing sixty-four.

Key knobs:

```yaml
behavior:
  temporal_window_seconds: 4.0
  alert_threshold: 85
  item_missing_timeout_seconds: 1.25
  min_person_track_seconds: 1.0
  min_item_association_seconds: 0.4
  cooldown_seconds: 30
  zone_only_risk_ceiling: 59      # see "Limitations"

recording:
  pre_event_seconds: 5
  post_event_seconds: 5
```

### Shelf zones

Because no COCO model recognizes general retail merchandise, the MVP grounds
interaction in configured geometry:

```yaml
zones:
  - id: shelf_001
    kind: shelf                   # shelf | high_value | basket | cart |
    name: Hair Care Left          # checkout | entrance | exclusion
    polygon:
      - [112, 148]
      - [604, 151]
      - [598, 702]
      - [107, 698]
```

Click zones out interactively and paste the YAML it prints:

```bash
python scripts/test_stream.py --source "$CAM_001_RTSP" --camera-id cam_001 \
    --pick-zone --zone-id shelf_001 --zone-kind shelf

# Then verify them against live video:
python scripts/test_stream.py --source "$CAM_001_RTSP" --camera-id cam_001 --zones
```

Full guidance, including how to avoid the common over-large-zone mistake:
[docs/ZONES.md](docs/ZONES.md).

---

## API

```bash
python -m aisleguardvision.main --headless --api
```

| Endpoint | Returns |
|---|---|
| `GET /health` | service and per-camera health *(unauthenticated, for probes)* |
| `GET /metrics` | pipeline metrics; `?format=prometheus` for text exposition |
| `GET /cameras` | all cameras, credential-sanitized |
| `GET /cameras/{id}` | one camera |
| `POST /cameras/{id}/enable` | enable a camera |
| `POST /cameras/{id}/disable` | disable a camera |
| `GET /events` | incidents (`?camera_id=`, `?limit=`, `?min_risk=`) |
| `GET /events/{id}` | full incident record with the complete risk arithmetic |

Interactive docs at `/docs`. Auth is a shared secret in `X-API-Key`, from
`AISLEGUARD_API_KEY`; empty disables auth and the server warns at startup.

Camera URLs are **never** returned — `source_label` is sanitized
(`rtsp://operator:***@10.0.0.5:554/stream`).

No frontend ships in Phase 1. A dashboard built over detection logic that has
not been validated against real footage is effort in the wrong place.

---

## Tests

```bash
pytest                      # 383 tests, no GPU or weights needed
pytest tests/test_scenarios.py -v      # the behavior guarantees
pytest --cov=aisleguardvision
```

Coverage includes: the nine behavior scenarios, geometry, risk arithmetic and
threat bands, state-machine ordering and benign branching, ByteTrack (including
occlusion recovery and FPS-independence), item occlusion ladder, hand/item and
pose/person association, circular buffer and bounded queue, camera reconnect
backoff and stall detection, incident recording, cooldown, alert dispatch, the
API, and architectural invariants such as the model-dependency firewall.

---

## Simulation

The most important development tool here. Behavior logic and model quality are
independent problems, and coupling them means every tuning change needs a GPU
and every regression is ambiguous.

```bash
python scripts/simulate_behavior.py
python scripts/simulate_behavior.py --scenario POSSIBLE_CONCEALMENT --verbose --timeline
python scripts/simulate_behavior.py --json
```

Scenarios emit scripted detections that run through the **production** tracker,
associator and behavior engine. Only the neural networks are absent.

```
  POSSIBLE_CONCEALMENT   [PASS]

  expected : ALERT
  actual   : ALERT
  peak risk: 100.0  (HIGH_RISK)
  peak state: REVIEW_ALERT

  State transitions:
      2.60s  track 1:  IDLE -> SHELF_INTERACTION
      2.95s  track 1:  SHELF_INTERACTION -> ITEM_ASSOCIATED
      3.75s  track 1:  ITEM_ASSOCIATED -> ITEM_REMOVED_FROM_SHELF
      4.45s  track 1:  ITEM_REMOVED_FROM_SHELF -> POSSIBLE_CONCEALMENT
      4.85s  track 1:  POSSIBLE_CONCEALMENT -> ITEM_OCCLUDED
      5.45s  track 1:  ITEM_OCCLUDED -> REVIEW_ALERT

  Evidence for concealment (at peak risk):
    + Right wrist interacted with shelf zone 'shelf_001' for 1.80s
    + Item 1 moved out of shelf zone 'shelf_001' while held
    + Right wrist showed a downward/inward motion profile (0.27 body-heights/s)
    + Right wrist associated with item 1 for 2.30s (confidence 0.80)
    + Right wrist moved from the shelf toward the right waist region (0.14 body-heights of travel)
    + Item 1 became occluded near the right waist region at (627, 360)
    + Item 1 remained unobserved for 1.25s

  Evidence against:
    (none)

  RESULT: ALERT
          An incident would be created for HUMAN REVIEW.
          This is NOT a determination that a theft occurred.
```

Available scenarios: `NORMAL_BROWSING`, `PHONE_INTERACTION`,
`POCKET_ADJUSTMENT`, `ITEM_PICKUP_RETURN`, `ITEM_TO_BASKET`, `ITEM_TO_CART`,
`TEMPORARY_OCCLUSION`, `POSSIBLE_CONCEALMENT`,
`CONCEALMENT_WITHOUT_ITEM_DETECTOR`.

---

## Benchmarking

```bash
python scripts/benchmark.py --source ./data/samples/store.mp4 --seconds 30
python scripts/benchmark.py --synthetic --cameras 1 4 8 16 32 --json bench.json
```

Measures decode FPS, detection FPS, pose FPS, end-to-end latency percentiles,
GPU memory, CPU usage, dropped frames and queue depth.

**This repository publishes no FPS or camera-count numbers.** Anyone quoting
"N cameras per GPU" without naming the model, resolution, detection rate, pose
rate and codec is guessing — the variables span more than an order of
magnitude. Run the benchmark on your hardware and use
[docs/SCALING.md](docs/SCALING.md) to size from the result.

---

## Limitations

**Stated plainly, because a loss-prevention system that overstates its
capability causes real harm.**

1. **No COCO model detects general retail merchandise.** Not wigs, hair
   bundles, gel, cosmetics, accessories, or most store inventory. Without a
   custom retail model the system runs zone-driven and its risk is **capped at
   59 against a threshold of 85 — it cannot alert.** This is deliberate: shelf
   geometry plus wrist kinematics is not sufficient grounds to ask a human to
   review a shopper. The config layer refuses to start if that invariant is
   broken.

   To lift it, train a YOLO-format model on your inventory and set
   `models.product`. Nothing else changes.

2. **Occlusion is fundamentally ambiguous from a single camera.** A shopper's
   own body occludes their hands constantly. The system models disappearance as
   a timed ladder and never treats it as a conclusion.

3. **Pose keypoints are noisy at retail camera angles and distances.** Low pose
   confidence is itself scored as negative evidence.

4. **Zones are per camera and per resolution.** Re-aiming a camera invalidates
   them.

5. **No cross-camera tracking.** A person leaving one view is an unrelated new
   track elsewhere. This is a capability limit and a privacy property.

6. **A 64-camera deployment needs work that is not written yet**: a worker
   supervisor, an aggregating central API, an incident database, and
   TensorRT/DeepStream backends. The interfaces exist; the implementations do
   not. See [docs/SCALING.md](docs/SCALING.md) for what is and is not there.

7. **Not validated against real store footage.** The behavior logic is verified
   against nine synthetic scenarios and 383 tests. That verifies the *logic*;
   it does not tell you the detection quality on your cameras, your lighting or
   your merchandise.

---

## Roadmap

**Jetson** ([docs/JETSON.md](docs/JETSON.md)) — JetPack 6, NVDEC hardware
decode, TensorRT FP16/INT8 engines, optional DeepStream. The abstraction is
built so the behavior engine is untouched by the move; a test enforces that
only one module imports the model framework.

**64 cameras** ([docs/SCALING.md](docs/SCALING.md)) — batched multi-camera
inference (the backend API is already shaped for it), GPU worker pools, process
isolation for failure domains, and two documented deployment architectures:
centralized GPU servers versus Jetson edge nodes, with the tradeoffs.

---

## Repository layout

```
config/            app.yaml, detection.yaml, cameras.yaml
src/aisleguardvision/
  core/            types, config, logging, metrics
  camera/          stream, worker, frame_buffer, manager
  inference/       backend, ultralytics_backend, detector, pose,
                   product_detector, scheduler, device
  tracking/        person_tracker (ByteTrack), item_tracker, association, kalman
  behavior/        engine, state_machine, evidence, risk, geometry, zones
  events/          recorder, dispatcher, cooldown, models
  visualization/   overlay
  api/             server, schemas, state
  simulation/      scenarios, harness
tests/             11 modules, 383 tests
scripts/           simulate_behavior, benchmark, test_stream, make_sample_video
docs/              ARCHITECTURE, SCALING, JETSON, FALSE_POSITIVES, ZONES
data/              incidents/, samples/
```

---

## Incidents on disk

```
data/incidents/cam_001/2026-09-12/
    3f2a....mp4     pre-event + event + post-event clip
    3f2a....jpg     annotated snapshot
    3f2a....json    full record, including the complete risk arithmetic
```

```json
{
  "event_id": "3f2a...",
  "camera_id": "cam_001",
  "person_id": 23,
  "risk_score": 91.2,
  "threat_level": "HIGH_RISK",
  "positive_evidence": [
    "Right wrist interacted with shelf zone 'Hair Care Left' for 0.40s",
    "Right wrist associated with item 45 for 1.2 seconds",
    "Right wrist moved from the shelf toward the right waist region",
    "Item 45 became occluded near the right waist region",
    "Item 45 remained unobserved for 1.70s"
  ],
  "negative_evidence": [],
  "evidence_breakdown": [ { "evidence": "...", "weight": 25, "points": 25.0 } ],
  "notice": "Possible concealment behavior detected - human review recommended. This is a behavioral risk signal, not a determination that a theft occurred..."
}
```

Only incidents are persisted — never the continuous stream.

---

## Security

- Credentials come from the environment; `config/*.yaml` contains only `${VAR}`
  references, and a test asserts no committed file contains a literal RTSP URL.
- Camera URLs are sanitized before reaching **any** log record, API response or
  on-screen overlay (`rtsp://operator:***@10.0.0.5:554/stream`).
- Webhook tokens come from the environment and are never logged; webhook URLs
  are stripped of userinfo and query strings before logging.
- The API key is compared in constant time.
- `.env` is git-ignored; `.env.example` contains only placeholders.
- Incident media stays on local disk unless you configure otherwise.
