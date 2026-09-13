# NVIDIA Jetson Migration

Target platforms: **Jetson AGX Orin** (64 GB / 32 GB) and **Jetson Orin NX**
(16 GB / 8 GB).

The goal of this document is that migrating changes the **inference layer
only**. The behavior engine, risk scoring, tracking, event pipeline and API are
untouched — that is the whole point of the `DetectorBackend` abstraction, and
it is enforced by a test (`test_only_one_module_owns_the_model_framework`).

---

## Why the current code is already Jetson-ready

| Design choice | Why it matters on Jetson |
|---|---|
| `DetectorBackend` returns normalized `Detection` objects | a TensorRT backend is a new subclass, not a rewrite |
| ByteTrack implemented in-repo with NumPy | no SciPy dependency — SciPy wheels on aarch64 are a recurring pain |
| Behavior engine imports no torch | you can develop and test on x86 with no Jetson present |
| All thresholds are durations | Orin NX runs lower FPS; nothing needs retuning for that |
| Conditional pose scheduling | the single biggest lever on a power-constrained device |
| `infer(list[...])` batch API | TensorRT engines are built for a fixed batch; the shape already matches |
| Bounded queues, drop-oldest | a thermally throttled device degrades gracefully instead of OOMing |

---

## Step 1 — JetPack and the base stack

Install **JetPack 6.x** (L4T r36.x), which provides CUDA 12.x, cuDNN 8.9+,
TensorRT 8.6+, VPI and the multimedia API.

```bash
sudo apt update && sudo apt install -y nvidia-jetpack
sudo nvpmodel -m 0        # maximum performance mode
sudo jetson_clocks        # pin clocks (see the thermal note below)
```

**Do not `pip install torch` from PyPI on Jetson.** The PyPI wheels are x86 or
CPU-only aarch64 builds with no CUDA support. Use NVIDIA's wheels:

```bash
# Check https://developer.download.nvidia.com/compute/redist/jp/ for your
# JetPack version, then:
pip install --no-cache-dir \
  --index-url https://developer.download.nvidia.com/compute/redist/jp/v60 \
  torch torchvision
```

Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from aisleguardvision.inference.device import select_device, log_device; \
           log_device(select_device('auto'))"
```

`DeviceInfo.is_jetson` detects Tegra-family device names and logs it. Nothing
in the pipeline branches on it behaviorally — it is there so a support engineer
reading logs knows what hardware produced them.

---

## Step 2 — Hardware video decode

**This is where the wins are, and where a naive port fails.** The default
OpenCV build decodes H.264/H.265 on the CPU. On Orin NX, 8 cameras of 1080p25
will saturate the CPU while the GPU sits idle.

Route decode to **NVDEC** via GStreamer:

```
rtspsrc location=<url> latency=100 protocols=tcp
  ! rtph264depay ! h264parse
  ! nvv4l2decoder                      ← NVDEC, zero CPU
  ! nvvidconv ! video/x-raw,format=BGRx
  ! videoconvert ! video/x-raw,format=BGR
  ! appsink drop=true max-buffers=1
```

Two options:

**(a) A GStreamer-enabled OpenCV.** Build OpenCV with `-DWITH_GSTREAMER=ON`,
then the existing `VideoStream` works unchanged if you pass the pipeline string
as the camera `source` — `cv2.VideoCapture` accepts a GStreamer pipeline
directly. Set it in `config/cameras.yaml`:

```yaml
cameras:
  - id: cam_001
    source: >-
      rtspsrc location=${CAM_001_RTSP} latency=100 protocols=tcp !
      rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv !
      video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR !
      appsink drop=true max-buffers=1
```

Credential handling is unchanged: the URL still comes from `${CAM_001_RTSP}`.

**(b) A DeepStream backend**, which keeps frames on the GPU end-to-end. See
Step 5.

The `nvvidconv → videoconvert → BGR` tail costs a device-to-host copy per
frame. Option (b) avoids it entirely; option (a) is far simpler and is the
right first move.

---

## Step 3 — TensorRT engines

Export once per device, per JetPack version. **Engines are not portable** — an
engine built on AGX Orin will not load on Orin NX, and one built under JetPack
6.0 may not load under 6.1. Build on the target.

```bash
yolo export model=yolo11n.pt format=engine half=True device=0 imgsz=640
yolo export model=yolo11n-pose.pt format=engine half=True device=0 imgsz=640
```

Point the config at the engines:

```yaml
# config/detection.yaml
models:
  backend: ultralytics        # Ultralytics loads .engine files directly
  detector: yolo11n.engine
  pose: yolo11n-pose.engine
  device: cuda:0
  fp16: true
```

The shipped `UltralyticsBackend` handles `.engine` weights with no code change,
because Ultralytics dispatches on the file extension. That gets you most of the
TensorRT benefit for the cost of an export command.

A dedicated `TensorRTBackend` (direct `trtexec`-built engines, no Ultralytics)
is worth writing only if you want to drop the Ultralytics dependency entirely
or need control over the execution context. The interface it must implement:

```python
class TensorRTBackend(DetectorBackend):
    def load(self) -> None: ...
    def _infer_batch(self, images: list[np.ndarray]) -> list[list[Detection]]: ...
    @property
    def capabilities(self) -> BackendCapabilities: ...

register_backend("tensorrt", TensorRTBackend)
```

Then `models.backend: tensorrt`. Nothing else in the repository changes.

### FP16 and INT8

FP16 is free accuracy-wise for detection and is on by default
(`models.fp16: true`); `DeviceInfo.supports_fp16` gates it to compute
capability ≥ 6.0, which every Orin satisfies (8.7).

INT8 needs a calibration dataset:

```bash
yolo export model=yolo11n.pt format=engine int8=True data=calibration.yaml
```

**Validate INT8 against real store footage before trusting it.** Quantization
error shows up as lost small-object detections — which, in this system, are
exactly the merchandise items the concealment sequence depends on. A detector
that silently stops seeing small items turns every genuine incident into a
`TEMPORARY_OCCLUSION` or `UNSTABLE_ITEM_TRACK`, and the system will quietly
stop alerting rather than fail loudly. Re-run
`scripts/simulate_behavior.py` for logic and real footage for detection quality.

---

## Step 4 — Power and thermal modes

```bash
sudo nvpmodel -q            # query current mode
sudo nvpmodel -m 0          # MAXN
sudo jetson_clocks          # disable dynamic frequency scaling
sudo tegrastats             # live GPU/CPU/thermal
```

`jetson_clocks` pins clocks at maximum, which removes latency jitter but raises
temperature. In a ceiling enclosure above a store aisle, thermal throttling is
a real operating condition, not an edge case — and when it happens, frame rate
drops.

**The system is built for that.** Every threshold is a duration, the scheduler
is time-based, and the queues drop stale frames rather than accumulating
latency. A throttled node processes fewer frames per second and reasons
correctly over the ones it gets. Watch `decode_fps` and `frames_dropped` in
`/metrics` to see it happening.

---

## Step 5 — DeepStream (optional, highest throughput)

DeepStream keeps frames on the GPU from NVDEC through inference, eliminating
every host copy. It is the right answer for 16+ cameras on one Orin.

```
nvurisrcbin ×N ──▶ nvstreammux ──▶ nvinfer (PGIE) ──▶ nvtracker ──▶ probe
      │                  │              │                  │          │
    NVDEC          batch on GPU      TensorRT          NvDCF      → Detection[]
```

Integration point: a `DeepStreamBackend` whose probe callback converts
`NvDsObjectMeta` into `Detection` objects and hands them to the existing
pipeline.

A deliberate note on the tracker: DeepStream's `nvtracker` (NvDCF) could
replace our ByteTrack. **Prefer keeping ours.** Track lifetimes here are
expressed in seconds and tuned for retail occlusion, the behavior engine
depends on that semantics, and it is covered by tests that run without any
NVIDIA hardware. Use `nvtracker` only if profiling shows our tracker is the
bottleneck, which is unlikely — it is NumPy over a handful of boxes.

---

## Suggested configuration per device

Starting points, **not measured claims**. Run `scripts/benchmark.py` on the
device and adjust.

### AGX Orin (64 GB) — up to ~16 cameras

```yaml
models:
  detector: yolo11s.engine
  pose: yolo11s-pose.engine
  fp16: true
inference:
  image_size: 640
  max_batch_size: 8
scheduler:
  detection_fps: 10
  pose_idle_fps: 2
  pose_active_fps: 10
  max_pose_targets: 6
recording:
  buffer_max_width: 1280
  buffer_max_frames: 400
```

### Orin NX (16 GB) — up to ~8 cameras

```yaml
models:
  detector: yolo11n.engine
  pose: yolo11n-pose.engine
  fp16: true
inference:
  image_size: 512
  max_batch_size: 4
scheduler:
  detection_fps: 6
  pose_idle_fps: 1
  pose_active_fps: 6
  max_pose_targets: 3
recording:
  buffer_max_width: 960
  buffer_max_frames: 250
```

### Orin NX (8 GB) — up to ~4 cameras

As above with `image_size: 416`, `detection_fps: 5`, `buffer_max_width: 640`.
At this tier, watch host RAM before GPU memory: the circular buffers are the
larger consumer.

---

## Migration checklist

- [ ] JetPack 6.x installed, `nvidia-jetpack` present
- [ ] NVIDIA torch wheels (**not** PyPI), `torch.cuda.is_available()` is True
- [ ] `select_device('auto')` reports `cuda:0` and `is_jetson` is True
- [ ] OpenCV built with GStreamer, or DeepStream in place
- [ ] Camera sources routed through `nvv4l2decoder`; `tegrastats` shows NVDEC active
- [ ] TensorRT engines exported **on this device**, FP16 enabled
- [ ] `nvpmodel -m 0` and thermal behaviour observed under sustained load
- [ ] `python -m pytest` passes on the device (it needs no GPU, so this is a
      clean check that the port did not break the logic)
- [ ] `python scripts/simulate_behavior.py` → 9/9 scenarios
- [ ] `python scripts/benchmark.py` run and its numbers recorded
- [ ] Real store footage validated, **especially if INT8 is enabled**
- [ ] Incident clips playable (check the codec — prefer the hardware encoder)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False | PyPI wheel installed | reinstall from NVIDIA's index |
| CPU pinned, GPU idle | software decode | route through `nvv4l2decoder` |
| Engine fails to load | built on different hardware or JetPack | rebuild on the target |
| FPS drops after ~10 minutes | thermal throttling | check `tegrastats`, improve airflow |
| Small items stop being detected | INT8 quantization error | revert to FP16 and re-validate |
| `cv2.VideoCapture` rejects the pipeline | OpenCV lacks GStreamer | rebuild with `-DWITH_GSTREAMER=ON` |
| Incident clips will not play | `mp4v` fallback | set `recording.codec` to a hardware-encoder FourCC |
| Out of memory at 8+ cameras | circular buffers | lower `buffer_max_width` / `buffer_max_frames` |
