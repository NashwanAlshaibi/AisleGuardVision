"""Hardware acceleration selection.

Priority is CUDA, then Apple MPS, then CPU. Nothing here assumes CUDA exists:
the MVP has to run on a developer laptop, and the whole non-inference stack has
to run with torch not installed at all.

Torch is imported lazily inside the functions so that importing this module --
which the config and API layers do transitively -- never drags in a 2 GB
framework.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved compute device and its capabilities."""

    #: Torch device string, e.g. "cuda:0", "mps", "cpu".
    device: str
    kind: str  # cuda | mps | cpu
    name: str = ""
    #: Whether FP16 inference is both supported and worth using here.
    supports_fp16: bool = False
    total_memory_mb: float = 0.0
    #: CUDA compute capability, when applicable, e.g. (8, 7) for Orin.
    compute_capability: tuple[int, int] | None = None

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def is_jetson(self) -> bool:
        """Heuristic: Jetson modules report a Tegra-family device name.

        Used only for logging and for choosing sensible defaults; nothing in
        the pipeline branches on it behaviorally.
        """
        lowered = self.name.lower()
        return any(token in lowered for token in ("orin", "xavier", "tegra", "jetson"))

    def describe(self) -> str:
        parts = [f"device={self.device}"]
        if self.name:
            parts.append(f"name={self.name}")
        if self.total_memory_mb:
            parts.append(f"memory={self.total_memory_mb:.0f}MB")
        parts.append(f"fp16={'yes' if self.supports_fp16 else 'no'}")
        return " ".join(parts)


def torch_available() -> bool:
    """Whether PyTorch can be imported in this process."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def select_device(preference: str = "auto", want_fp16: bool = True) -> DeviceInfo:
    """Resolve a device from a preference string.

    ``preference`` accepts ``auto``, ``cpu``, ``mps``, ``cuda`` or an explicit
    ``cuda:N``. Anything unavailable falls back with a warning rather than
    raising -- a store must keep processing video on CPU rather than stop
    because a driver update broke CUDA.
    """
    if not torch_available():
        logger.warning(
            "PyTorch is not installed; live inference is unavailable "
            "(install the 'inference' extra). Behavior engine and simulation still run."
        )
        return DeviceInfo(device="cpu", kind="cpu", name="cpu (torch not installed)")

    import torch

    preference = (preference or "auto").strip().lower()

    def _cuda(index: int = 0) -> DeviceInfo:
        properties = torch.cuda.get_device_properties(index)
        capability = (properties.major, properties.minor)
        # FP16 is a real win from Pascal (6.x) onwards; below that it is
        # emulated and can be slower than FP32.
        fp16 = want_fp16 and capability >= (6, 0)
        return DeviceInfo(
            device=f"cuda:{index}",
            kind="cuda",
            name=properties.name,
            supports_fp16=fp16,
            total_memory_mb=properties.total_memory / (1024 * 1024),
            compute_capability=capability,
        )

    if preference.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return _cpu_device()
        index = 0
        if ":" in preference:
            try:
                index = int(preference.split(":", 1)[1])
            except ValueError:
                logger.warning("malformed CUDA device %r; using cuda:0", preference)
        if index >= torch.cuda.device_count():
            logger.warning(
                "CUDA device index out of range; using cuda:0",
                extra={"fields": {"requested": index, "available": torch.cuda.device_count()}},
            )
            index = 0
        return _cuda(index)

    if preference == "mps":
        if _mps_available(torch):
            return DeviceInfo(device="mps", kind="mps", name="Apple Metal", supports_fp16=False)
        logger.warning("MPS requested but unavailable; falling back to CPU")
        return _cpu_device()

    if preference == "cpu":
        return _cpu_device()

    # auto
    if torch.cuda.is_available():
        return _cuda(0)
    if _mps_available(torch):
        # FP16 on MPS is inconsistent across torch versions; leave it off.
        return DeviceInfo(device="mps", kind="mps", name="Apple Metal", supports_fp16=False)
    return _cpu_device()


def _mps_available(torch) -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _cpu_device() -> DeviceInfo:
    import platform

    return DeviceInfo(
        device="cpu",
        kind="cpu",
        name=platform.processor() or platform.machine() or "cpu",
        supports_fp16=False,
    )


def log_device(info: DeviceInfo) -> None:
    """Emit the startup line recording what the system chose."""
    logger.info(
        "compute device selected",
        extra={
            "fields": {
                "device": info.device,
                "kind": info.kind,
                "name": info.name,
                "fp16": info.supports_fp16,
                "memory_mb": round(info.total_memory_mb),
                "jetson": info.is_jetson,
            }
        },
    )


def cuda_memory_mb() -> tuple[float, float]:
    """Return ``(allocated_mb, reserved_mb)``; zeros when CUDA is unavailable.

    Used by the benchmark script so GPU memory figures come from measurement
    rather than estimation.
    """
    if not torch_available():
        return 0.0, 0.0
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    mib = 1024 * 1024
    return (
        torch.cuda.memory_allocated() / mib,
        torch.cuda.memory_reserved() / mib,
    )
