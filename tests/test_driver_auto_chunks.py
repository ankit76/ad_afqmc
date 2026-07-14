from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from trot import driver
from trot.prop.types import QmcParams


@dataclass(frozen=True)
class _FakeMemoryStats:
    temp_size_in_bytes: int
    argument_size_in_bytes: int = 0
    output_size_in_bytes: int = 0
    alias_size_in_bytes: int = 0


class _FakeCompiled:
    def __init__(self, estimated_bytes: int | None):
        self.estimated_bytes = estimated_bytes

    def memory_analysis(self):
        if self.estimated_bytes is None:
            return None
        return _FakeMemoryStats(temp_size_in_bytes=self.estimated_bytes)


class _FakeRunBlocks:
    def __init__(self, n_chunks: int, estimated_bytes: int | None):
        self.n_chunks = n_chunks
        self.estimated_bytes = estimated_bytes
        self.lower_kwargs: dict[str, Any] | None = None

    def lower(self, state, **kwargs):
        del state
        self.lower_kwargs = kwargs
        return self

    def compile(self):
        return _FakeCompiled(self.estimated_bytes)


def _select(monkeypatch, params: QmcParams, estimates: dict[int, int | None]):
    built: list[_FakeRunBlocks] = []

    def fake_make_run_blocks(*, params, **kwargs):
        del kwargs
        run_blocks = _FakeRunBlocks(params.n_chunks, estimates[params.n_chunks])
        built.append(run_blocks)
        return run_blocks

    monkeypatch.setattr(driver, "make_run_blocks", fake_make_run_blocks)
    monkeypatch.setattr(driver, "_device_memory_limit_bytes", lambda state: 1000)
    selected, run_blocks = driver._make_run_blocks_with_auto_chunks(
        block_fn=None,  # type: ignore[arg-type]
        sys=None,  # type: ignore[arg-type]
        params=params,
        trial_ops=None,  # type: ignore[arg-type]
        meas_ops=None,  # type: ignore[arg-type]
        prop_ops=None,  # type: ignore[arg-type]
        state=None,  # type: ignore[arg-type]
        ham_data=None,
        trial_data=None,
        meas_ctx=None,
        prop_ctx=None,
        observable_names=(),
    )
    return selected, run_blocks, built


def test_compiled_memory_bytes_subtracts_aliases():
    compiled = _FakeCompiled(estimated_bytes=100)
    stats = _FakeMemoryStats(
        temp_size_in_bytes=100,
        argument_size_in_bytes=40,
        output_size_in_bytes=20,
        alias_size_in_bytes=10,
    )
    compiled.memory_analysis = lambda: stats  # type: ignore[method-assign]
    assert driver._compiled_memory_bytes(compiled) == 150


def test_auto_chunks_reuses_first_candidate_when_it_fits(monkeypatch):
    params = QmcParams(
        n_walkers=8,
        n_chunks=1,
        auto_n_chunks=True,
        n_eql_blocks=50,
        n_blocks=100,
    )
    selected, run_blocks, built = _select(monkeypatch, params, {1: 700})

    assert selected.n_chunks == 1
    assert run_blocks is built[0]
    assert built[0].lower_kwargs is not None
    assert built[0].lower_kwargs["n_blocks"] == 10


def test_auto_chunks_increases_until_compiler_estimate_fits(monkeypatch):
    params = QmcParams(n_walkers=8, n_chunks=1, auto_n_chunks=True)
    selected, run_blocks, built = _select(monkeypatch, params, {1: 1200, 2: 700})

    assert [candidate.n_chunks for candidate in built] == [1, 2]
    assert selected.n_chunks == 2
    assert run_blocks is built[-1]


def test_auto_chunks_keeps_compiled_candidate_without_memory_analysis(monkeypatch):
    params = QmcParams(n_walkers=8, n_chunks=2, auto_n_chunks=True)
    selected, run_blocks, built = _select(monkeypatch, params, {2: None})

    assert selected.n_chunks == 2
    assert run_blocks is built[0]


def test_auto_chunks_raises_when_one_walker_exceeds_budget(monkeypatch):
    params = QmcParams(n_walkers=2, n_chunks=1, auto_n_chunks=True)
    with pytest.raises(MemoryError, match="one-walker chunk"):
        _select(monkeypatch, params, {1: 1200, 2: 900})
