from __future__ import annotations

from dataclasses import dataclass

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import build_meas_ctx as build_pt2_dense_ctx
from trot.meas.pt2ccsd import energy_kernel_rw_rh as pt2_dense_components
from trot.meas.ptccsd import build_ptccsd_meas_ctx
from trot.meas.ptccsd import energy_components_pt_rw_rh as pt_dense_inverse_components
from trot.meas.ptccsd import energy_pt_rw_rh as pt_dense_energy
from trot.meas.ptccsd import force_bias_pt_rw_rh as pt_dense_force_bias
from trot.meas.ptccsd_modes import (
    _components_pt_thouless_rw_rh_full,
    _force_bias_pt_thouless_rw_rh_full,
    build_ptccsd_mode_meas_ctx,
    build_ptccsd_thouless_mode_meas_ctx,
    components_pt_rw_rh,
    components_pt_thouless_rw_rh,
    energy_pt_rw_rh as pt_mode_energy,
    energy_pt_thouless_rw_rh as pt_thouless_mode_energy,
    force_bias_pt_rw_rh as pt_mode_force_bias,
    force_bias_pt_thouless_rw_rh as pt_thouless_mode_force_bias,
    inverse_guide_components_pt_rw_rh as pt_mode_inverse_components,
    make_ptccsd_mode_meas_ops,
    make_ptccsd_thouless_mode_meas_ops,
)
from trot.meas.ptccsd_thouless import build_ptccsd_thouless_meas_ctx
from trot.meas.ptccsd_thouless import energy_pt_rw_rh as pt_thouless_dense_energy
from trot.meas.ptccsd_thouless import force_bias_pt_rw_rh as pt_thouless_dense_force_bias
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.ptccsd import PtccsdTrial, overlap_pt_r as pt_dense_overlap
from trot.trial.ptccsd_modes import (
    PtccsdModeTrial,
    PtccsdThoulessModeTrial,
    decompose_t2_modes,
    overlap_pt_r as pt_mode_overlap,
    overlap_ptccsd_thouless_r as pt_thouless_mode_overlap,
)
from trot.trial.ptccsd_thouless import (
    PtccsdThoulessTrial,
    overlap_ptccsd_thouless_r as pt_thouless_dense_overlap,
)


@dataclass(frozen=True)
class PtCases:
    sys: System
    ham: HamChol
    walkers: jax.Array
    dense: PtccsdTrial
    mode: PtccsdModeTrial
    dense_thouless: PtccsdThoulessTrial
    mode_thouless: PtccsdThoulessModeTrial


@pytest.fixture(scope="module")
def pt_cases() -> PtCases:
    rng = np.random.default_rng(8127)
    nocc, nvir, nchol = 2, 3, 5
    norb = nocc + nvir

    t1 = 0.08 * rng.normal(size=(nocc, nvir))
    raw = 0.04 * rng.normal(size=(nocc, nvir, nocc, nvir))
    t2 = 0.5 * (raw + raw.transpose(2, 3, 0, 1))
    eigenvalues, modes = decompose_t2_modes(t2)

    h1_raw = rng.normal(size=(norb, norb))
    h1 = 0.5 * (h1_raw + h1_raw.T)
    chol_raw = rng.normal(size=(nchol, norb, norb))
    chol = 0.5 * (chol_raw + chol_raw.transpose(0, 2, 1))
    ham = HamChol(
        h0=jnp.asarray(0.37),
        h1=jnp.asarray(h1),
        chol=jnp.asarray(chol),
        basis="restricted",
    )

    walkers = []
    for _ in range(4):
        walker = np.eye(norb, nocc) + 0.12 * rng.normal(size=(norb, nocc))
        walker = walker + 0.07j * rng.normal(size=(norb, nocc))
        walkers.append(walker)
    walkers_array = jnp.asarray(np.stack(walkers))

    mo_t = jnp.vstack([jnp.eye(nocc), jnp.asarray(t1).T])
    dense = PtccsdTrial(t1=jnp.asarray(t1), t2=jnp.asarray(t2))
    mode = PtccsdModeTrial(
        t1=jnp.asarray(t1),
        eigenvalues=jnp.asarray(eigenvalues),
        modes=jnp.asarray(modes),
    )
    dense_thouless = PtccsdThoulessTrial(mo_t=mo_t, t2=jnp.asarray(t2))
    mode_thouless = PtccsdThoulessModeTrial(
        mo_t=mo_t,
        eigenvalues=jnp.asarray(eigenvalues),
        modes=jnp.asarray(modes),
    )
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    return PtCases(
        sys=sys,
        ham=ham,
        walkers=walkers_array,
        dense=dense,
        mode=mode,
        dense_thouless=dense_thouless,
        mode_thouless=mode_thouless,
    )


@pytest.mark.parametrize("n_mode_chunks", [1, 2, 5])
def test_dense_and_full_rank_pt_modes_match(pt_cases: PtCases, n_mode_chunks: int):
    case = pt_cases
    dense_ctx = build_ptccsd_meas_ctx(case.ham, case.dense)
    mode_ctx = build_ptccsd_mode_meas_ctx(
        case.ham,
        case.mode,
        n_mode_chunks=n_mode_chunks,
    )

    dense_overlap = jax.vmap(pt_dense_overlap, in_axes=(0, None))(case.walkers, case.dense)
    mode_overlap = jax.vmap(pt_mode_overlap, in_axes=(0, None))(case.walkers, case.mode)
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=2.0e-11, atol=2.0e-11)

    dense_fb = jax.vmap(pt_dense_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense
    )
    mode_fb = jax.vmap(pt_mode_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mode_ctx, case.mode
    )
    np.testing.assert_allclose(mode_fb, dense_fb, rtol=2.0e-10, atol=2.0e-10)

    dense_energy = jax.vmap(pt_dense_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense
    )
    mode_energy = jax.vmap(pt_mode_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mode_ctx, case.mode
    )
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=3.0e-10, atol=3.0e-10)

    dense_components = jax.vmap(
        pt_dense_inverse_components,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, dense_ctx, case.dense)
    mode_components = jax.vmap(
        pt_mode_inverse_components,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode)
    np.testing.assert_allclose(mode_components, dense_components, rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("n_mode_chunks", [1, 2, 5])
def test_dense_and_full_rank_thouless_modes_match(pt_cases: PtCases, n_mode_chunks: int):
    case = pt_cases
    dense_ctx = build_ptccsd_thouless_meas_ctx(case.ham, case.dense_thouless)
    mode_ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=n_mode_chunks,
    )

    dense_overlap = jax.vmap(pt_thouless_dense_overlap, in_axes=(0, None))(
        case.walkers, case.dense_thouless
    )
    mode_overlap = jax.vmap(pt_thouless_mode_overlap, in_axes=(0, None))(
        case.walkers, case.mode_thouless
    )
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=2.0e-11, atol=2.0e-11)

    dense_fb = jax.vmap(pt_thouless_dense_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense_thouless
    )
    mode_fb = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode_thouless)
    np.testing.assert_allclose(mode_fb, dense_fb, rtol=3.0e-10, atol=3.0e-10)

    dense_energy = jax.vmap(pt_thouless_dense_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, case.dense_thouless
    )
    mode_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode_thouless)
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=3.0e-10, atol=3.0e-10)


def test_dense_pt2_components_and_full_rank_modes_match(pt_cases: PtCases):
    case = pt_cases
    dense_pt2 = Pt2ccsdTrial(
        mo_t=case.dense_thouless.mo_t,
        t2=case.dense_thouless.t2,
    )
    dense_ctx = build_pt2_dense_ctx(case.ham, dense_pt2)
    mode_ctx = build_ptccsd_thouless_mode_meas_ctx(case.ham, case.mode_thouless)

    dense_components = jax.vmap(pt2_dense_components, in_axes=(0, None, None, None))(
        case.walkers, case.ham, dense_ctx, dense_pt2
    )
    mode_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mode_ctx, case.mode_thouless)
    np.testing.assert_allclose(mode_components, dense_components, rtol=3.0e-10, atol=3.0e-10)

    weights = jnp.asarray([0.7, 1.2, 0.9, 1.4])
    dense_avg = jnp.sum(weights[:, None] * dense_components, axis=0) / jnp.sum(weights)
    mode_avg = jnp.sum(weights[:, None] * mode_components, axis=0) / jnp.sum(weights)

    def combine(components):
        theta, electronic_0, h_t = components
        return case.ham.h0 + electronic_0 + h_t - theta * electronic_0

    np.testing.assert_allclose(combine(mode_avg), combine(dense_avg), rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("n_mode_chunks", [1, 2, 5])
def test_half_green_thouless_kernels_match_full_green_oracle(
    pt_cases: PtCases,
    n_mode_chunks: int,
    memory_mode: str,
):
    case = pt_cases
    ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=n_mode_chunks,
        memory_mode=memory_mode,
    )

    full_force_bias = jax.vmap(
        _force_bias_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    half_force_bias = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    np.testing.assert_allclose(half_force_bias, full_force_bias, rtol=3.0e-10, atol=3.0e-10)

    full_components = jax.vmap(
        _components_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    half_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    np.testing.assert_allclose(half_components, full_components, rtol=3.0e-10, atol=3.0e-10)

    half_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, case.mode_thouless)
    full_energy = (
        case.ham.h0
        + full_components[:, 1]
        + full_components[:, 2]
        - full_components[:, 0] * full_components[:, 1]
    )
    np.testing.assert_allclose(half_energy, full_energy, rtol=3.0e-10, atol=3.0e-10)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_half_green_thouless_kernels_match_full_oracle_in_complex_gauge(
    pt_cases: PtCases,
    memory_mode: str,
):
    """Exercise the identities without assuming the occupied block of C is I."""

    case = pt_cases
    occupied_gauge = jnp.asarray(
        [[1.1 + 0.2j, -0.1 + 0.05j], [0.08 - 0.04j, 0.9 - 0.15j]]
    )
    trial = PtccsdThoulessModeTrial(
        mo_t=case.mode_thouless.mo_t @ occupied_gauge,
        eigenvalues=case.mode_thouless.eigenvalues,
        modes=case.mode_thouless.modes,
    )
    ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        trial,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )

    full_force_bias = jax.vmap(
        _force_bias_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    half_force_bias = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    np.testing.assert_allclose(half_force_bias, full_force_bias, rtol=3.0e-10, atol=3.0e-10)

    full_components = jax.vmap(
        _components_pt_thouless_rw_rh_full,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    half_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, ctx, trial)
    np.testing.assert_allclose(half_components, full_components, rtol=3.0e-10, atol=3.0e-10)


def test_mode_meas_factories_expose_guide_kernels(pt_cases: PtCases):
    pt_ops = make_ptccsd_mode_meas_ops(pt_cases.sys)
    thouless_ops = make_ptccsd_thouless_mode_meas_ops(pt_cases.sys)
    assert pt_ops.has_kernel(k_force_bias) and pt_ops.has_kernel(k_energy)
    assert thouless_ops.has_kernel(k_force_bias) and thouless_ops.has_kernel(k_energy)
    assert thouless_ops.build_meas_ctx(pt_cases.ham, pt_cases.mode_thouless).memory_mode == "high"


def test_ptccsd_mode_mixed_precision_matches_cisd_accuracy_policy(pt_cases: PtCases):
    case = pt_cases
    mixed_trial = PtccsdModeTrial(
        t1=case.mode.t1,
        eigenvalues=case.mode.eigenvalues,
        modes=case.mode.modes.astype(jnp.float32),
    )
    full_ctx = build_ptccsd_mode_meas_ctx(case.ham, case.mode, n_mode_chunks=2)
    mixed_ops = make_ptccsd_mode_meas_ops(case.sys, n_mode_chunks=2)
    mixed_ctx = mixed_ops.build_meas_ctx(case.ham, mixed_trial)

    assert mixed_ctx.cfg.mixed_real_dtype == jnp.float32
    assert mixed_ctx.cfg.mixed_complex_dtype == jnp.complex64

    full_fb = jax.vmap(pt_mode_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, full_ctx, case.mode
    )
    mixed_fb = jax.vmap(pt_mode_force_bias, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mixed_ctx, mixed_trial
    )
    full_energy = jax.vmap(pt_mode_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, full_ctx, case.mode
    )
    mixed_energy = jax.vmap(pt_mode_energy, in_axes=(0, None, None, None))(
        case.walkers, case.ham, mixed_ctx, mixed_trial
    )
    full_overlap = jax.vmap(pt_mode_overlap, in_axes=(0, None))(case.walkers, case.mode)
    mixed_overlap = jax.vmap(pt_mode_overlap, in_axes=(0, None))(
        case.walkers, mixed_trial
    )

    overlap_error = float(
        jnp.linalg.norm(mixed_overlap - full_overlap) / jnp.linalg.norm(full_overlap)
    )
    fb_error = float(jnp.linalg.norm(mixed_fb - full_fb) / jnp.linalg.norm(full_fb))
    energy_error = float(jnp.max(jnp.abs(mixed_energy - full_energy)))
    assert overlap_error < 1.0e-5
    assert fb_error < 2.0e-5
    assert energy_error < 2.0e-4


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_ptccsd_thouless_mixed_precision_matches_cisd_accuracy_policy(
    pt_cases: PtCases,
    memory_mode: str,
):
    case = pt_cases
    mixed_trial = PtccsdThoulessModeTrial(
        mo_t=case.mode_thouless.mo_t,
        eigenvalues=case.mode_thouless.eigenvalues,
        modes=case.mode_thouless.modes.astype(jnp.float32),
    )
    full_ctx = build_ptccsd_thouless_mode_meas_ctx(
        case.ham,
        case.mode_thouless,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )
    mixed_ops = make_ptccsd_thouless_mode_meas_ops(
        case.sys,
        n_mode_chunks=2,
        memory_mode=memory_mode,
    )
    mixed_ctx = mixed_ops.build_meas_ctx(case.ham, mixed_trial)

    assert mixed_ctx.cfg.mixed_real_dtype == jnp.float32
    assert mixed_ctx.cfg.mixed_complex_dtype == jnp.complex64

    full_fb = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_ctx, case.mode_thouless)
    mixed_fb = jax.vmap(
        pt_thouless_mode_force_bias,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mixed_ctx, mixed_trial)
    full_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_ctx, case.mode_thouless)
    mixed_components = jax.vmap(
        components_pt_thouless_rw_rh,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mixed_ctx, mixed_trial)
    full_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, full_ctx, case.mode_thouless)
    mixed_energy = jax.vmap(
        pt_thouless_mode_energy,
        in_axes=(0, None, None, None),
    )(case.walkers, case.ham, mixed_ctx, mixed_trial)

    fb_error = float(jnp.linalg.norm(mixed_fb - full_fb) / jnp.linalg.norm(full_fb))
    component_error = float(jnp.max(jnp.abs(mixed_components - full_components)))
    energy_error = float(jnp.max(jnp.abs(mixed_energy - full_energy)))
    assert fb_error < 2.0e-5
    assert component_error < 2.0e-4
    assert energy_error < 2.0e-4
