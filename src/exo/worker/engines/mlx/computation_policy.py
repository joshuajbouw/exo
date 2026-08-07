"""Operator policy for disposable computation projections."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG


@dataclass(frozen=True, slots=True)
class ComputationRetentionPolicy:
    """Optional limits supplied by the owning resource authority.

    ``None`` means that no blanket limit was supplied. Exo still evicts an
    ancestor projection after publishing its verified successor; these values
    govern independent lineage leaves and future cold-chain materialization.
    """

    projection_budget_bytes: int | None = None
    maximum_cold_reconstruction_seconds: float | None = None

    @classmethod
    def from_environment(cls) -> ComputationRetentionPolicy:
        return cls(
            projection_budget_bytes=_optional_positive_int(
                "EXO_COMPUTATION_PROJECTION_BUDGET_BYTES"
            ),
            maximum_cold_reconstruction_seconds=_optional_positive_float(
                "EXO_COMPUTATION_MAX_COLD_RECONSTRUCTION_SECONDS"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProjectionReclamation:
    paths: tuple[Path, ...] = ()
    bytes_reclaimed: int = 0
    bytes_retained: int = 0


def reclaim_projection_budget(
    directory: Path,
    budget_bytes: int | None,
) -> ProjectionReclamation:
    """Evict oldest disposable projections until an explicit budget fits."""
    candidates: list[tuple[int, int, Path]] = []
    retained = 0
    for path in directory.glob("*.safetensors"):
        try:
            status = path.lstat()
        except FileNotFoundError:
            continue
        if not S_ISREG(status.st_mode) or path.is_symlink():
            path.unlink(missing_ok=True)
            continue
        retained += status.st_size
        candidates.append((status.st_mtime_ns, status.st_size, path))
    if budget_bytes is None or retained <= budget_bytes:
        return ProjectionReclamation(bytes_retained=retained)
    removed: list[Path] = []
    reclaimed = 0
    for _, size, path in sorted(candidates):
        if retained <= budget_bytes:
            break
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue
        removed.append(path)
        reclaimed += size
        retained -= size
    return ProjectionReclamation(tuple(removed), reclaimed, retained)


def _optional_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    parsed = float(value)
    if not parsed > 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed
