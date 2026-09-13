"""Architectural invariants.

These guard properties that are easy to break with a single convenient import
and expensive to discover later. Each one corresponds to a claim made in
docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "aisleguardvision"

#: The only module permitted to import torch or ultralytics at module level.
MODEL_FRAMEWORK_OWNERS = {"ultralytics_backend.py"}

#: Modules allowed to import a framework *lazily*, inside a function body, so
#: that importing them costs nothing when the framework is absent.
LAZY_FRAMEWORK_USERS = {"device.py", "product_detector.py"}

#: The CLI entry point legitimately writes to stdout/stderr: the banner, the
#: benchmark tables and configuration errors are its user interface.
PRINT_ALLOWED = {"main.py"}

#: Documentation and help-text examples. A real credential would not match one
#: of these placeholder userinfo pairs.
CREDENTIAL_PLACEHOLDERS = (
    "user:pass",
    "bob:hunter2",
    "operator:hunter2",
    "viewer:CHANGE_ME",
    "u:p@",
    "<user>:<password>",
)

#: Packages that must stay free of deep-learning frameworks so that the
#: behavior logic, the test suite and the simulator run without a GPU.
FRAMEWORK_FREE_PACKAGES = ("behavior", "tracking", "events", "core", "api", "utils")

#: Packages that must additionally stay free of OpenCV, so behavior logic can
#: run in any environment and is never entangled with rendering.
CV2_FREE_PACKAGES = ("behavior", "tracking", "core", "utils")


def imported_modules(path: Path, module_level_only: bool = False) -> set[str]:
    """Top-level module names imported by a Python file.

    ``module_level_only`` restricts the scan to imports in the file body, which
    is what determines whether merely importing the module pulls the dependency
    in. An import inside a function is a deliberate lazy load.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body if module_level_only else list(ast.walk(tree))
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Try) and module_level_only:
            # A guarded `try: import torch` is still a module-level import.
            nodes.extend(node.body)
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def python_files(package: str) -> list[Path]:
    return sorted((PACKAGE / package).rglob("*.py"))


@pytest.mark.parametrize("package", FRAMEWORK_FREE_PACKAGES)
def test_packages_do_not_import_deep_learning_frameworks(package):
    """The dependency firewall.

    If this fails, the behavior engine can no longer be developed or tested
    without a GPU, and swapping the inference backend stops being safe.
    """
    offenders = []
    for path in python_files(package):
        if path.name in MODEL_FRAMEWORK_OWNERS:
            continue
        leaked = imported_modules(path) & {"torch", "torchvision", "ultralytics"}
        if leaked:
            offenders.append(f"{path.relative_to(SRC)} imports {sorted(leaked)}")
    assert not offenders, "deep-learning framework leaked past the backend:\n  " + "\n  ".join(
        offenders
    )


@pytest.mark.parametrize("package", CV2_FREE_PACKAGES)
def test_decision_packages_do_not_import_opencv(package):
    offenders = [
        str(path.relative_to(SRC))
        for path in python_files(package)
        if "cv2" in imported_modules(path)
    ]
    assert not offenders, f"cv2 leaked into decision logic: {offenders}"


def test_only_one_module_owns_the_model_framework():
    """Keeping the coupling surface to a single file is what makes a TensorRT
    or Triton backend a local change."""
    owners = sorted(
        path.relative_to(SRC)
        for path in PACKAGE.rglob("*.py")
        if imported_modules(path, module_level_only=True) & {"torch", "ultralytics"}
    )
    assert [p.name for p in owners] == ["ultralytics_backend.py"], owners


def test_lazy_framework_users_import_inside_functions_only():
    """device.py may touch torch, but only lazily, so that importing it on a
    machine without torch stays free."""
    for name in LAZY_FRAMEWORK_USERS:
        matches = list(PACKAGE.rglob(name))
        if not matches:
            continue
        path = matches[0]
        assert not (imported_modules(path, module_level_only=True) & {"torch", "ultralytics"}), (
            f"{name} must import the framework lazily, not at module level"
        )


def test_importing_the_pipeline_does_not_load_torch():
    """Static analysis proves no import statement; this proves no *runtime*
    import either, which is what keeps startup fast and CI GPU-free."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import aisleguardvision.pipeline, aisleguardvision.api.server; "
            "print(sorted(m for m in ('torch','ultralytics') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"loaded at import time: {result.stdout}"


def test_no_print_statements_in_production_modules():
    """Production modules log; they do not print. Scripts are exempt."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if path.name in PRINT_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"print() in production code: {offenders}"


def test_behavior_engine_does_not_depend_on_the_inference_package():
    """The behavior engine must never see a backend, a model or a raw frame."""
    offenders = []
    for path in python_files("behavior"):
        source = path.read_text(encoding="utf-8")
        if "from ..inference" in source or "import aisleguardvision.inference" in source:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"behavior depends on inference: {offenders}"


def test_no_hardcoded_credentials_or_camera_addresses():
    """Credentials and camera addresses come from the environment."""
    suspicious = []
    for path in PACKAGE.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or ">>>" in stripped:
                continue
            if "rtsp://" not in stripped:
                continue
            after = stripped.split("rtsp://", 1)[1].split()[0]
            if "@" not in after:
                continue
            # Documentation and help-text examples use obvious placeholders; a
            # real leaked credential would not.
            if any(marker in stripped for marker in CREDENTIAL_PLACEHOLDERS):
                continue
            suspicious.append(f"{path.relative_to(SRC)}:{number} -> {stripped[:70]}")
    assert not suspicious, f"possible hard-coded credential: {suspicious}"


def test_shipped_config_contains_no_credentials():
    config_dir = SRC.parent / "config"
    for path in config_dir.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "rtsp://" in line and "${" not in line and not line.strip().startswith("#"):
                pytest.fail(f"{path.name} has a literal RTSP URL: {line.strip()}")


def test_every_evidence_type_has_a_factory_or_is_deliberately_internal():
    """A new evidence type without a factory would produce an alert line with
    no human-readable justification."""
    from aisleguardvision.behavior import evidence as ev
    from aisleguardvision.core.types import EvidenceType

    source = (PACKAGE / "behavior" / "evidence.py").read_text(encoding="utf-8")
    missing = [
        member.value for member in EvidenceType if f"EvidenceType.{member.name}" not in source
    ]
    assert not missing, f"evidence types with no factory: {missing}"
    assert hasattr(ev, "shelf_interaction")


def test_public_docs_exist():
    docs = SRC.parent / "docs"
    for name in ("ARCHITECTURE.md", "SCALING.md", "JETSON.md", "FALSE_POSITIVES.md", "ZONES.md"):
        assert (docs / name).exists(), f"missing {name}"
    assert (SRC.parent / "README.md").exists()
