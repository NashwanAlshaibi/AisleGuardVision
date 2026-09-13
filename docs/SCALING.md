# Scaling AisleGuard Vision — 1 to 64 cameras

## Read this first

**This document contains no performance claims.** It contains the arithmetic
you need to size a deployment, and instructions for measuring your own numbers.

Anyone who tells you "this GPU handles N cameras" without naming the model, the
resolution, the detection rate, the pose rate and the codec is guessing. The
variables span more than an order of magnitude.

Measure your hardware:

```bash
python scripts/benchmark.py --source ./data/samples/store.mp4 --seconds 30
python scripts/benchmark.py --synthetic --cameras 1 4 8 16 32 --seconds 20 --json bench.json
```

`--cameras N` replays the workload through N independent pipelines. That is an
estimate of *compute* headroom only. It does not model RTSP bandwidth, NVDEC
decoder slots, per-process GPU context overhead, or your network. Treat it as
an upper bound, then validate against real cameras.

---

## The arithmetic that actually matters

64 cameras at 30 FPS is **1,920 frames per second**. No GPU runs a detector on
all of them, and no amount of optimization changes that. The design answer is
to decouple three rates:

| Rate | Typical | Why |
|---|---|---|
| Camera decode | 25–30 FPS | whatever the stream delivers |
| YOLO detection | **10 FPS** | Kalman tracking interpolates between detections |
| Pose estimation | **2–10 FPS, conditional** | only for people near merchandise |

That changes the workload from 1,920 detector-frames/s to **640** — and pose,
the more expensive model, runs only on the handful of shoppers actually at a
shelf face rather than on every person in every frame.

### Sizing formula

```
detector_load  = cameras × detection_fps
pose_load      = cameras × pose_fps × P(person near merchandise)
gpu_frames_sec = detector_load + pose_load

cameras_per_gpu = measured_model_fps / (detection_fps + pose_fps × P)
```

`measured_model_fps` comes from `scripts/benchmark.py` on **your** hardware.
`P` is the fraction of frames containing someone the scheduler considers
relevant — in a typical aisle this is well under 0.3, and the scheduler's
`pose_relevance_distance_ratio` controls it directly.

Worked example, with `measured_model_fps` as the variable you supply:

```
detection_fps = 10, pose_fps = 8, P = 0.25
per-camera GPU load = 10 + 8×0.25 = 12 model-frames/second

cameras_per_gpu = measured_model_fps / 12
```

If your benchmark reports 240 model-frames/second, that is 20 cameras/GPU,
before leaving headroom. Apply a **60–70% utilization target** — a GPU pinned
at 100% has no margin for a detection burst when a tour group walks through.

### What else binds, in the order you will hit it

1. **Decode, not inference.** 64 × 1080p25 H.264 is roughly 1,600 Mbit/s of
   decode work. On a dGPU this must go to NVDEC (~2 chips' worth on most
   cards); on CPU it will saturate cores long before the GPU is busy.
2. **Network.** 64 × 4 Mbit/s ≈ 256 Mbit/s sustained, multicast-unfriendly,
   on the same LAN as the POS. Budget a dedicated VLAN.
3. **Host RAM.** The circular buffer is the big consumer: 5 s pre-roll at
   1280×720×3 bytes × 25 FPS ≈ 165 MB/camera uncompressed. `buffer_max_width`
   (default 1280) and `buffer_max_frames` (default 600) bound it — at 64
   cameras that is ~10 GB, so tune both down or accept the RAM.
4. **GPU VRAM.** One shared model instance per worker process, not per camera.
   Sharing is the entire reason `CameraPipeline` takes an injected detector.
5. **Disk.** Incidents only — never the continuous stream. A 64-camera store
   retaining raw video writes on the order of a terabyte a day.

---

## Scaling stages

### 1 camera — the MVP as it ships

```
Camera ──▶ decode thread ──▶ queue ──▶ single analysis loop ──▶ incidents
```

One process. `main.py` as written. No changes needed.

### 4–8 cameras — one process, shared models

```
cam_1 ─┐
cam_2 ─┼─▶ decode threads ─▶ queues ─┐
cam_3 ─┤                             ├─▶ one analysis loop ─▶ shared detector
cam_4 ─┘                             ┘        (round-robin over cameras)
```

Already supported: `main.py` iterates every camera per loop pass, and all
pipelines share one `PersonDetector`. Decode threads genuinely parallelize
because OpenCV/FFmpeg release the GIL.

**Turn batching on here.** `inference.max_batch_size: 8` coalesces frames from
several cameras into one GPU invocation — `detect_batch()` already accepts
exactly that, and `BatchAccumulator` in `inference/scheduler.py` does the
grouping.

### 8–16 cameras — batched inference, tuned rates

```yaml
# config/detection.yaml
inference:
  max_batch_size: 8
  batch_timeout_ms: 8          # latency/throughput knob
scheduler:
  detection_fps: 8
  pose_idle_fps: 1
  pose_active_fps: 8
  max_pose_targets: 4
```

Batching is where the GPU actually starts being used efficiently: a YOLO11n
forward pass on one 640×640 frame leaves most of the card idle, and eight
frames cost far less than 8× one frame.

### 16–32 cameras — multiple worker processes

Move from threads to **processes**, for failure isolation rather than for
parallelism:

```
             ┌──────────────────────────────┐
cam 1-8  ───▶│ worker process 1 (GPU 0)     │─┐
             └──────────────────────────────┘ │
             ┌──────────────────────────────┐ │
cam 9-16 ───▶│ worker process 2 (GPU 0)     │─┼─▶ incidents/ (shared volume)
             └──────────────────────────────┘ │
             ┌──────────────────────────────┐ │        │
cam 17-24───▶│ worker process 3 (GPU 1)     │─┘        ▼
             └──────────────────────────────┘   Central API / dashboard
```

Each process runs the `main.py` loop over its own camera subset:

```bash
python -m aisleguardvision.main --headless --camera cam_001 --camera cam_002 ... &
python -m aisleguardvision.main --headless --camera cam_009 ... &
```

A segfault in an FFmpeg decoder — which does happen with malformed RTSP —
takes down 8 cameras, not 64. That is the whole argument for processes.

### 32–64 cameras — the two deployment architectures

---

## Architecture A — centralized GPU servers

```
        64 IP cameras (dedicated VLAN)
                    │
       ┌────────────┼────────────┬────────────┐
       ▼            ▼            ▼            ▼
  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ GPU srv1│  │ GPU srv2│  │ GPU srv3│  │ GPU srv4│
  │ cam 1-16│  │cam 17-32│  │cam 33-48│  │cam 49-64│
  │ NVDEC + │  │         │  │         │  │         │
  │ TensorRT│  │         │  │         │  │         │
  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
       └────────────┴─────┬──────┴────────────┘
                          ▼
              ┌───────────────────────┐
              │  Central AisleGuard   │
              │  API + incident store │
              │  + dashboard          │
              └───────────────────────┘
```

**Advantages**

- Cheapest per camera: one A2/L4/A10 amortizes across 16 cameras.
- One place to update models; one place to tune risk weights.
- NVDEC on a server card handles many more streams than a Jetson's decoder.
- Multi-GPU is straightforward — assign worker processes to `cuda:N`.

**Disadvantages**

- All 64 streams traverse the network continuously. 256+ Mbit/s sustained,
  and the store keeps paying for it whether anything is happening or not.
- A server failure blackouts 16 cameras at once.
- Network partition = total loss of coverage. The store keeps recording to its
  NVR, but AisleGuard sees nothing.
- Higher latency: encode → network → decode → infer.

**Choose this when** the store has good internal networking, a server room,
and you want the lowest hardware cost and the simplest model-update story.

---

## Architecture B — Jetson edge nodes

```
Camera group A (16) ──▶ ┌──────────────────┐
                        │ Jetson AGX Orin  │──┐
                        │ NVDEC + TensorRT │  │
                        └──────────────────┘  │
Camera group B (16) ──▶ ┌──────────────────┐  │
                        │ Jetson AGX Orin  │──┤
                        └──────────────────┘  ├──▶ Central dashboard
Camera group C (16) ──▶ ┌──────────────────┐  │    (events + clips only)
                        │ Jetson Orin NX   │──┤
                        └──────────────────┘  │
Camera group D (16) ──▶ ┌──────────────────┐  │
                        │ Jetson Orin NX   │──┘
                        └──────────────────┘
```

**Advantages**

- Video never leaves the aisle. Only incident metadata and short clips cross
  the network — kilobytes per incident instead of megabytes per second.
- Failure domain is 16 cameras and physically localized.
- Runs during a WAN outage; events queue and sync later.
- Meaningfully better privacy posture, which matters for this product category:
  raw footage of shoppers stays on a device in the store.
- Low latency — no network hop before inference.

**Disadvantages**

- Higher hardware cost per camera.
- Orin NX decoder and GPU are both weaker than a server card; expect to run
  yolo11n, lower `detection_fps`, and lean harder on conditional pose.
- Model updates must be pushed to N physical devices.
- TensorRT engines are **not portable** — they are built per device, per
  JetPack version. See [JETSON.md](JETSON.md).

**Choose this when** privacy posture matters, the WAN is unreliable, or the
store cannot host a server.

---

## Comparison

| | A: GPU servers | B: Jetson edge |
|---|---|---|
| Hardware cost / camera | lower | higher |
| Network load | continuous, high | bursty, tiny |
| Failure domain | 16 cameras, remote | 16 cameras, local |
| Survives WAN outage | no | yes |
| Raw video leaves the store floor | yes | no |
| Model update | one host | N devices |
| Latency | network + inference | inference only |
| Best for | cost-sensitive, good LAN | privacy-sensitive, poor WAN |

A realistic large deployment is **hybrid**: Jetson nodes on the high-value
aisles where privacy and latency matter most, a GPU server for the general
floor, and one central API.

---

## What the code already supports

| Capability | Status |
|---|---|
| Batched multi-camera inference | `infer(list[np.ndarray]) -> list[list[Detection]]`, `detect_batch()`, `BatchAccumulator` |
| Shared model across cameras | `CameraPipeline` takes an injected detector |
| Per-camera rate overrides | `cameras[].target_inference_fps` |
| Conditional pose | `scheduler.adaptive_pose`, `pose_only_for_relevant_people` |
| Per-camera failure isolation | `CameraManager`, one decode thread + one stream per camera |
| Backend swap (TensorRT/Triton/DeepStream) | `DetectorBackend` + `register_backend` |
| Health and metrics per camera | `GET /health`, `GET /metrics`, labelled instruments |
| Bounded memory under overload | bounded queues, capped buffers, drop-oldest |

## What a 64-camera deployment still needs

Honestly stated, because none of this is written yet:

1. **A worker supervisor** — process launch, camera assignment, restart on
   crash. Today you run N `main.py` processes yourself.
2. **A central API that aggregates workers.** The current API serves one
   process. A central instance needs to fan out to workers or read a shared
   incident store.
3. **An incident database.** The filesystem store is inspectable and adequate
   for one store; multi-store review wants Postgres.
4. **TensorRT/DeepStream backends.** The interface is ready; the
   implementations are not written.
5. **The dashboard.** Deliberately not built in Phase 1 — a UI over detection
   logic that has not been validated against real footage is effort in the
   wrong place.
6. **Zone calibration tooling at scale.** Calibrating 64 cameras by clicking
   polygons is a day of work; `scripts/test_stream.py --pick-zone` is the
   single-camera version of what needs to become a proper tool.

---

## Health monitoring at scale

At 64 cameras something is always broken. The metrics that tell you what:

| Metric | Watch for |
|---|---|
| `decode_fps{camera_id}` | at or near 0 → camera down or stalled |
| `camera_reconnects{camera_id}` | climbing → flaky PoE or network |
| `frames_dropped{camera_id}` | climbing → the worker is behind; reduce rates |
| `queue_depth{camera_id}` | persistently at max → same |
| `inference_latency_ms` p95 | rising → GPU saturated |
| `active_tracks{camera_id}` | 0 during trading hours → check the view |
| `clip_write_errors` | non-zero → disk full or codec missing |
| `webhook_failures` | non-zero → alerts are not reaching anyone |

`GET /health` returns `degraded` rather than `unhealthy` when some cameras are
down: the service is still doing useful work on the ones that are up, and
paging someone at 2am for one flaky camera trains them to ignore the pager.
