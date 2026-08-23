from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np


DENSE_MEMORY_BUDGET_FRACTION = 0.8
DENSE_WORKSPACE_MATRIX_FACTOR = 6


def _read_memory_counter(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cgroup_memory_available_bytes() -> int | None:
    """Return the narrowest cgroup memory headroom containing this process."""

    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None

    cgroup_root = Path("/sys/fs/cgroup")
    candidates: list[int] = []
    for membership in memberships:
        fields = membership.split(":", maxsplit=2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if controllers == "":
            directory = cgroup_root / relative.lstrip("/")
            limit_name = "memory.max"
            usage_name = "memory.current"
        elif "memory" in controllers.split(","):
            directory = cgroup_root / "memory" / relative.lstrip("/")
            limit_name = "memory.limit_in_bytes"
            usage_name = "memory.usage_in_bytes"
        else:
            continue

        while directory == cgroup_root or cgroup_root in directory.parents:
            limit = _read_memory_counter(directory / limit_name)
            usage = _read_memory_counter(directory / usage_name)
            # Very large cgroup-v1 limits conventionally mean "unlimited".
            if limit is not None and limit < 1 << 60 and usage is not None:
                candidates.append(max(0, limit - usage))
            if directory == cgroup_root:
                break
            directory = directory.parent

    return min(candidates) if candidates else None


def _linux_memory_available_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return None


def _macos_memory_available_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    page_size_match = re.search(r"page size of (\d+) bytes", result.stdout)
    if page_size_match is None:
        return None
    page_size = int(page_size_match.group(1))
    pages = 0
    available_names = {"Pages free", "Pages inactive", "Pages speculative"}
    for line in result.stdout.splitlines():
        name, separator, value = line.partition(":")
        if separator and name in available_names:
            pages += int(value.strip().rstrip("."))
    return pages * page_size if pages else None


def available_host_memory_bytes() -> int | None:
    """Estimate memory presently available to this process.

    On Linux the result respects both the host-wide ``MemAvailable`` value and
    any enclosing Slurm/container cgroup limit. On macOS it uses reclaimable
    pages reported by ``vm_stat``.
    """

    candidates = [
        value
        for value in (
            _linux_memory_available_bytes(),
            _cgroup_memory_available_bytes(),
            _macos_memory_available_bytes() if os.uname().sysname == "Darwin" else None,
        )
        if value is not None
    ]
    return min(candidates) if candidates else None


def dense_factorization_memory_bytes(dimension: int) -> int:
    """Conservative incremental peak-memory estimate for ``numpy.linalg.eigh``."""

    matrix_bytes = dimension * dimension * np.dtype(np.float64).itemsize
    return DENSE_WORKSPACE_MATRIX_FACTOR * matrix_bytes


def auto_dense_is_memory_safe(dimension: int) -> tuple[bool, int, int | None]:
    estimate = dense_factorization_memory_bytes(dimension)
    available = available_host_memory_bytes()
    safe = available is None or estimate <= DENSE_MEMORY_BUDGET_FRACTION * available
    return safe, estimate, available


def format_dense_memory_selection(dimension: int) -> tuple[bool, str]:
    safe, estimate, available = auto_dense_is_memory_safe(dimension)
    available_text = f"{available / 1024**3:.3f} GiB" if available is not None else "unknown"
    message = (
        f"peak_increment={estimate / 1024**3:.3f} GiB, "
        f"available={available_text}, "
        f"budget_fraction={DENSE_MEMORY_BUDGET_FRACTION:.2f}"
    )
    return safe, message
