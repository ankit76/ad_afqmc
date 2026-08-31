import numpy as np

from trot.trial import mode_factorization


def test_lp64_dense_selection_rejects_dsyevd_workspace_overflow(monkeypatch):
    monkeypatch.setattr(
        mode_factorization,
        "available_host_memory_bytes",
        lambda: 1 << 60,
    )
    monkeypatch.setattr(mode_factorization, "numpy_linalg_uses_ilp64", lambda: False)

    safe_below_limit, message_below_limit = (
        mode_factorization.format_dense_memory_selection(32_766)
    )
    safe_above_limit, message_above_limit = (
        mode_factorization.format_dense_memory_selection(32_767)
    )

    assert safe_below_limit
    assert "lapack_integer=LP64" in message_below_limit
    assert "workspace_index_safe=True" in message_below_limit
    assert not safe_above_limit
    assert "lwork=2147549181" in message_above_limit
    assert "workspace_index_safe=False" in message_above_limit
    assert mode_factorization.dense_eigh_lwork_elements(32_767) > np.iinfo(np.int32).max


def test_ilp64_dense_selection_allows_large_dsyevd_workspace(monkeypatch):
    monkeypatch.setattr(
        mode_factorization,
        "available_host_memory_bytes",
        lambda: 1 << 60,
    )
    monkeypatch.setattr(mode_factorization, "numpy_linalg_uses_ilp64", lambda: True)

    safe, message = mode_factorization.format_dense_memory_selection(34_125)

    assert safe
    assert "lapack_integer=ILP64" in message
    assert "lwork=2329236001" in message
    assert "workspace_index_safe=True" in message
