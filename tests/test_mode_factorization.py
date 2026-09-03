import numpy as np
import pytest

from trot.trial import mode_factorization


def test_lp64_dense_selection_accepts_large_dsyevr_workspace(monkeypatch):
    monkeypatch.setattr(
        mode_factorization,
        "available_host_memory_bytes",
        lambda: 1 << 60,
    )
    monkeypatch.setattr(mode_factorization, "scipy_linalg_uses_ilp64", lambda: False)

    dimension = 33_928
    safe, message = mode_factorization.format_dense_memory_selection(dimension)

    assert safe
    assert "lapack_driver=DSYEVR" in message
    assert "lapack_integer=LP64" in message
    assert f"lwork={26 * dimension}" in message
    assert f"liwork={10 * dimension}" in message
    assert "workspace_index_safe=True" in message


def test_lp64_dense_selection_rejects_dsyevr_workspace_overflow(monkeypatch):
    monkeypatch.setattr(
        mode_factorization,
        "available_host_memory_bytes",
        lambda: 1 << 200,
    )
    monkeypatch.setattr(mode_factorization, "scipy_linalg_uses_ilp64", lambda: False)

    dimension = np.iinfo(np.int32).max // 26 + 1
    safe, message = mode_factorization.format_dense_memory_selection(dimension)

    assert not safe
    assert "lapack_driver=DSYEVR" in message
    assert "lapack_integer=LP64" in message
    assert "workspace_index_safe=False" in message


def test_dense_symmetric_eigh_uses_evr_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.asfortranarray([[2.0, 0.5], [0.5, 1.0]])
    expected_values = np.array([0.7928932188134524, 2.2071067811865475])
    expected_vectors = np.eye(2)
    captured: dict[str, object] = {}

    def fake_eigh(argument: np.ndarray, **kwargs: object):
        captured["matrix"] = argument
        captured.update(kwargs)
        return expected_values, expected_vectors

    monkeypatch.setattr(mode_factorization.scipy_linalg, "eigh", fake_eigh)

    values, vectors = mode_factorization.dense_symmetric_eigh(matrix)

    assert captured == {
        "matrix": matrix,
        "driver": "evr",
        "overwrite_a": True,
        "check_finite": False,
    }
    assert values is expected_values
    assert vectors is expected_vectors
