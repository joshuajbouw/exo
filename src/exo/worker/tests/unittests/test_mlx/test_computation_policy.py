from pathlib import Path

import pytest

from exo.worker.engines.mlx.computation_policy import (
    ComputationRetentionPolicy,
    reclaim_projection_budget,
)


def test_projection_budget_evicts_oldest_without_a_fixed_default(tmp_path: Path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"a" * 7)
    second.write_bytes(b"b" * 11)
    first.touch()
    second.touch()

    unlimited = reclaim_projection_budget(tmp_path, None)
    assert unlimited.bytes_retained == 18
    assert not unlimited.paths

    reclaimed = reclaim_projection_budget(tmp_path, 11)
    assert reclaimed.paths == (first,)
    assert reclaimed.bytes_reclaimed == 7
    assert reclaimed.bytes_retained == 11
    assert second.exists()


def test_projection_budget_never_turns_eviction_failure_into_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    projection = tmp_path / "only.safetensors"
    projection.write_bytes(b"data")

    monkeypatch.setattr(
        Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )
    result = reclaim_projection_budget(tmp_path, 1)

    assert not result.paths
    assert result.bytes_retained == 4


def test_retention_policy_rejects_non_positive_operator_limits(monkeypatch):
    monkeypatch.setenv("EXO_COMPUTATION_PROJECTION_BUDGET_BYTES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        ComputationRetentionPolicy.from_environment()
