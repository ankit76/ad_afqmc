from __future__ import annotations

from trot import config

config.configure_once(use_gpu=False)

import numpy as np

from trot.core.system import System
from trot.trial.ptuccsd_modes import (
    make_split_ptuccsd_thouless_mode_trial_data,
)


def _pyscf_layout(block: np.ndarray) -> np.ndarray:
    return block.transpose(0, 2, 1, 3)


def test_split_builder_factors_native_space_then_pads_modes() -> None:
    rng = np.random.default_rng(84_221)
    runtime_norb = 7
    norb_spin = (5, 7)
    noa, nob = 2, 3
    native_nvir = (3, 4)
    runtime_nvir = (5, 4)

    dim_a = noa * native_nvir[0]
    dim_b = nob * native_nvir[1]
    kaa = rng.standard_normal((dim_a, dim_a))
    kaa = 0.5 * (kaa + kaa.T)
    kab = rng.standard_normal((dim_a, dim_b))
    kbb = rng.standard_normal((dim_b, dim_b))
    kbb = 0.5 * (kbb + kbb.T)
    t2aa = kaa.reshape(noa, native_nvir[0], noa, native_nvir[0])
    t2ab = kab.reshape(noa, native_nvir[0], nob, native_nvir[1])
    t2bb = kbb.reshape(nob, native_nvir[1], nob, native_nvir[1])

    t2aa_padded = np.full((noa, runtime_nvir[0], noa, runtime_nvir[0]), 99.0)
    t2ab_padded = np.full((noa, runtime_nvir[0], nob, runtime_nvir[1]), 99.0)
    t2bb_padded = np.array(t2bb, copy=True)
    t2aa_padded[:, : native_nvir[0], :, : native_nvir[0]] = t2aa
    t2ab_padded[:, : native_nvir[0], :, : native_nvir[1]] = t2ab

    data = {
        "t1a": np.pad(
            rng.standard_normal((noa, native_nvir[0])),
            ((0, 0), (0, runtime_nvir[0] - native_nvir[0])),
        ),
        "t1b": rng.standard_normal((nob, native_nvir[1])),
        "t2aa": _pyscf_layout(t2aa_padded),
        "t2ab": _pyscf_layout(t2ab_padded),
        "t2bb": _pyscf_layout(t2bb_padded),
        "mo_coeff_b": np.eye(runtime_norb),
    }
    sys = System(runtime_norb, (noa, nob), walker_kind="unrestricted")
    trial = make_split_ptuccsd_thouless_mode_trial_data(
        data,
        norb_spin,
        sys,
        discarded_norm_target=0.0,
        mixed_precision=False,
        mode_solver="dense",
        verbose=False,
    )

    expected_mo_t_a = np.vstack(
        (np.eye(noa), np.asarray(data["t1a"]).T)
    )
    expected_mo_t_b = np.vstack(
        (np.eye(nob), np.asarray(data["t1b"]).T)
    )
    np.testing.assert_allclose(trial.mo_t_a, expected_mo_t_a)
    np.testing.assert_allclose(trial.mo_t_b, expected_mo_t_b)
    kernel = np.asarray(trial.modes).T @ (
        np.asarray(trial.eigenvalues)[:, None] * np.asarray(trial.modes)
    )
    runtime_dim_a = noa * runtime_nvir[0]
    reconstructed_aa = kernel[:runtime_dim_a, :runtime_dim_a].reshape(
        noa, runtime_nvir[0], noa, runtime_nvir[0]
    )
    reconstructed_ab = kernel[:runtime_dim_a, runtime_dim_a:].reshape(
        noa, runtime_nvir[0], nob, runtime_nvir[1]
    )
    reconstructed_bb = kernel[runtime_dim_a:, runtime_dim_a:].reshape(
        nob, runtime_nvir[1], nob, runtime_nvir[1]
    )

    assert trial.pair_dim == (noa * runtime_nvir[0], nob * runtime_nvir[1])
    np.testing.assert_allclose(
        reconstructed_aa[:, : native_nvir[0], :, : native_nvir[0]],
        t2aa,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        reconstructed_ab[:, : native_nvir[0], :, : native_nvir[1]],
        t2ab,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(reconstructed_bb, t2bb, atol=2.0e-13)
    np.testing.assert_array_equal(reconstructed_aa[:, native_nvir[0] :, :, :], 0.0)
    np.testing.assert_array_equal(reconstructed_aa[:, :, :, native_nvir[0] :], 0.0)
    np.testing.assert_array_equal(reconstructed_ab[:, native_nvir[0] :, :, :], 0.0)


def test_split_builder_truncates_by_discarded_norm() -> None:
    norb = 3
    noa = nob = 1
    t2aa = np.diag([4.0, 3.0]).reshape(1, 2, 1, 2)
    t2ab = np.zeros((1, 2, 1, 2))
    t2bb = np.diag([2.0, 1.0]).reshape(1, 2, 1, 2)
    data = {
        "t1a": np.zeros((1, 2)),
        "t1b": np.zeros((1, 2)),
        "t2aa": _pyscf_layout(t2aa),
        "t2ab": _pyscf_layout(t2ab),
        "t2bb": _pyscf_layout(t2bb),
        "mo_coeff_b": np.eye(norb),
    }
    trial = make_split_ptuccsd_thouless_mode_trial_data(
        data,
        (norb, norb),
        System(norb, (noa, nob), walker_kind="unrestricted"),
        discarded_norm_target=0.25,
        mixed_precision=False,
        mode_solver="dense",
        verbose=False,
    )

    assert trial.mode_rank == 3
    retained_norm = np.linalg.norm(np.asarray(trial.eigenvalues))
    full_norm = np.linalg.norm([4.0, 3.0, 2.0, 1.0])
    discarded_fraction = np.sqrt(full_norm**2 - retained_norm**2) / full_norm
    assert discarded_fraction <= 0.25
    assert np.sqrt(2.0**2 + 1.0**2) / full_norm > 0.25
