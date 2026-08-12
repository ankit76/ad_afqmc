from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import ao2mo, cc, gto, mcscf, scf

from trot import testing
from trot.afqmc import Afqmc
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.cisd import (
    CisdMeasCfg,
    build_meas_ctx,
    energy_kernel_rw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_rw_rh_high,
    force_bias_kernel_rw_rh_high_complex,
    force_bias_kernel_rw_rh_high_realimag,
    force_bias_kernel_rw_rh_low,
    get_cisd_meas_cfg,
    make_cisd_meas_ops,
    rdm1_kernel_rw,
)
from trot.prop.chol_afqmc_ops import _build_prop_ctx
from trot.prop.types import QmcParams
from trot.runtime_layout import (
    CisdHostRuntimeLayout,
    _build_cisd_meas_ctx_from_host,
    _build_restricted_prop_ctx_from_host,
    _make_ham_data,
)
from trot.setup import setup as setup_job
from trot.staging import _freeze_core_from_mo_cholesky
from trot.trial.cisd import (
    CisdTrial,
    make_cisd_trial_data,
    make_cisd_trial_ops,
    overlap_r,
    overlap_r_high,
    overlap_r_high_complex,
    overlap_r_high_realimag,
    overlap_r_low,
)


def _exact_cholesky_from_mo_eri(eri: np.ndarray) -> np.ndarray:
    nmo = int(eri.shape[0])
    eri_pair = np.asarray(eri).reshape(nmo * nmo, nmo * nmo)
    eri_pair = 0.5 * (eri_pair + eri_pair.T)
    eig, vec = np.linalg.eigh(eri_pair)
    keep = eig > 1.0e-12
    chol = np.sqrt(eig[keep])[:, None] * vec[:, keep].T
    return chol.reshape(-1, nmo, nmo)


def _make_cisd_trial(
    key,
    norb: int,
    nocc: int,
    *,
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
) -> CisdTrial:
    """
    Random CISD coefficients in the MO basis where the reference occupies [0..nocc-1].

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    nvir = norb - nocc
    k1, k2 = jax.random.split(key)

    ci1 = scale_ci1 * jax.random.normal(k1, (nocc, nvir), dtype=dtype)
    ci2 = scale_ci2 * jax.random.normal(k2, (nocc, nvir, nocc, nvir), dtype=dtype)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    # Use high precision for the "testing" dtypes so the manual kernel is not
    # artificially noisy from float32/complex64 paths.
    return CisdTrial(ci1=ci1, ci2=ci2)


def _pad_cisd_amplitudes(
    ci1: jax.Array,
    ci2: jax.Array,
    *,
    nocc_t_core: int,
    nvir_t_outer: int,
) -> tuple[jax.Array, jax.Array]:
    nocc_act, nvir_act = ci1.shape
    nocc_full = nocc_t_core + nocc_act
    nvir_full = nvir_act + nvir_t_outer

    ci1_full = jnp.zeros((nocc_full, nvir_full), dtype=ci1.dtype)
    ci1_full = ci1_full.at[nocc_t_core:, :nvir_act].set(ci1)

    ci2_full = jnp.zeros((nocc_full, nvir_full, nocc_full, nvir_full), dtype=ci2.dtype)
    ci2_full = ci2_full.at[nocc_t_core:, :nvir_act, nocc_t_core:, :nvir_act].set(ci2)
    return ci1_full, ci2_full


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_active_space_matches_zero_padded_full_space(nocc_t_core, nvir_t_outer):
    key = jax.random.PRNGKey(321)
    k1, k2, k_ham, k_w = jax.random.split(key, 4)

    nocc_full = 4
    nvir_full = 6
    nocc_act = nocc_full - nocc_t_core
    nvir_act = nvir_full - nvir_t_outer

    ci1 = 0.05 * jax.random.normal(k1, (nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.02 * jax.random.normal(k2, (nocc_act, nvir_act, nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    trial_active = CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ci1_full, ci2_full = _pad_cisd_amplitudes(
        ci1,
        ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    trial_full = CisdTrial(ci1=ci1_full, ci2=ci2_full)

    norb_full = nocc_full + nvir_full
    ham = testing.make_random_ham_chol(k_ham, norb=norb_full, n_chol=8, basis="restricted")
    ctx_active = build_meas_ctx(ham, trial_active)
    ctx_full = build_meas_ctx(ham, trial_full)

    for i in range(1):
        walker = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb_full, nocc_full, mix=0.25
        )
        o_active = overlap_r(walker, trial_active)
        o_full = overlap_r(walker, trial_full)
        assert jnp.allclose(o_active, o_full, rtol=1e-12, atol=1e-12), (o_active, o_full)

        fb_active = force_bias_kernel_rw_rh(walker, ham, ctx_active, trial_active)
        fb_full = force_bias_kernel_rw_rh(walker, ham, ctx_full, trial_full)
        assert jnp.allclose(fb_active, fb_full, rtol=1e-12, atol=1e-12), (fb_active, fb_full)

        e_active = energy_kernel_rw_rh(walker, ham, ctx_active, trial_active)
        e_full = energy_kernel_rw_rh(walker, ham, ctx_full, trial_full)
        assert jnp.allclose(e_active, e_full, rtol=1e-12, atol=1e-12), (e_active, e_full)

        dm_active = rdm1_kernel_rw(walker, ham, ctx_active, trial_active)
        dm_full = rdm1_kernel_rw(walker, ham, ctx_full, trial_full)
        assert jnp.allclose(dm_active, dm_full, rtol=1e-12, atol=1e-12), (dm_active, dm_full)


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_overlap_memory_modes_match(nocc_t_core, nvir_t_outer):
    key = jax.random.PRNGKey(911)
    k1, k2, k_w = jax.random.split(key, 3)

    nocc_full = 4
    nvir_full = 6
    nocc_act = nocc_full - nocc_t_core
    nvir_act = nvir_full - nvir_t_outer

    ci1 = 0.05 * jax.random.normal(k1, (nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.02 * jax.random.normal(k2, (nocc_act, nvir_act, nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    trial = CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )

    norb_full = nocc_full + nvir_full
    for i in range(1):
        walker = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb_full, nocc_full, mix=0.25
        )
        o_high = overlap_r_high(walker, trial)
        o_high_complex = overlap_r_high_complex(walker, trial)
        o_high_realimag = overlap_r_high_realimag(walker, trial)
        o_low = overlap_r_low(walker, trial)
        assert jnp.allclose(o_high_complex, o_high, rtol=1e-12, atol=1e-12), (
            o_high_complex,
            o_high,
        )
        assert jnp.allclose(o_high_realimag, o_high, rtol=1e-12, atol=1e-12), (
            o_high_realimag,
            o_high,
        )
        assert jnp.allclose(o_low, o_high, rtol=1e-12, atol=1e-12), (o_low, o_high)


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_force_bias_high_variants_match(nocc_t_core, nvir_t_outer):
    key = jax.random.PRNGKey(919)
    k1, k2, k_ham, k_w = jax.random.split(key, 4)

    nocc_full = 4
    nvir_full = 6
    nocc_act = nocc_full - nocc_t_core
    nvir_act = nvir_full - nvir_t_outer

    ci1 = 0.05 * jax.random.normal(k1, (nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.02 * jax.random.normal(k2, (nocc_act, nvir_act, nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    trial = CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ham = testing.make_random_ham_chol(
        k_ham, norb=nocc_full + nvir_full, n_chol=8, basis="restricted"
    )
    ctx = build_meas_ctx(ham, trial)

    for i in range(1):
        walker = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), nocc_full + nvir_full, nocc_full, mix=0.25
        )
        fb_high_complex = force_bias_kernel_rw_rh_high_complex(walker, ham, ctx, trial)
        fb_high = force_bias_kernel_rw_rh_high(walker, ham, ctx, trial)
        fb_high_realimag = force_bias_kernel_rw_rh_high_realimag(walker, ham, ctx, trial)
        fb_low = force_bias_kernel_rw_rh_low(walker, ham, ctx, trial)
        assert jnp.allclose(fb_high_complex, fb_high, rtol=1e-12, atol=1e-12), (
            fb_high_complex,
            fb_high,
        )
        assert jnp.allclose(fb_high_realimag, fb_high, rtol=1e-12, atol=1e-12), (
            fb_high_realimag,
            fb_high,
        )
        assert jnp.allclose(fb_low, fb_high, rtol=1e-12, atol=1e-12), (fb_low, fb_high)


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_force_bias_variants_match_mixed_precision(nocc_t_core, nvir_t_outer):
    key = jax.random.PRNGKey(929)
    k1, k2, k_ham, k_w = jax.random.split(key, 4)

    nocc_full = 4
    nvir_full = 6
    nocc_act = nocc_full - nocc_t_core
    nvir_act = nvir_full - nvir_t_outer

    ci1 = 0.05 * jax.random.normal(k1, (nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.02 * jax.random.normal(k2, (nocc_act, nvir_act, nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    trial = CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ham = testing.make_random_ham_chol(
        k_ham, norb=nocc_full + nvir_full, n_chol=8, basis="restricted"
    )
    cfg = CisdMeasCfg(
        mixed_real_dtype=jnp.float32,
        mixed_complex_dtype=jnp.complex64,
        mixed_real_dtype_testing=jnp.float32,
        mixed_complex_dtype_testing=jnp.complex64,
    )
    ctx = build_meas_ctx(ham, trial, cfg=cfg)

    for i in range(1):
        walker = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), nocc_full + nvir_full, nocc_full, mix=0.25
        )
        fb_high_complex = force_bias_kernel_rw_rh_high_complex(walker, ham, ctx, trial)
        fb_high = force_bias_kernel_rw_rh_high(walker, ham, ctx, trial)
        fb_low = force_bias_kernel_rw_rh_low(walker, ham, ctx, trial)
        assert jnp.allclose(fb_high, fb_high_complex, rtol=2e-5, atol=2e-5), (
            fb_high,
            fb_high_complex,
        )
        assert jnp.allclose(fb_low, fb_high, rtol=2e-5, atol=2e-5), (fb_low, fb_high)


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_energy_memory_modes_match(nocc_t_core, nvir_t_outer):
    key = jax.random.PRNGKey(939)
    k1, k2, k_ham, k_w = jax.random.split(key, 4)

    nocc_full = 4
    nvir_full = 6
    nocc_act = nocc_full - nocc_t_core
    nvir_act = nvir_full - nvir_t_outer

    ci1 = 0.05 * jax.random.normal(k1, (nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.02 * jax.random.normal(k2, (nocc_act, nvir_act, nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    trial = CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ham = testing.make_random_ham_chol(
        k_ham, norb=nocc_full + nvir_full, n_chol=8, basis="restricted"
    )
    cfg_high = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    cfg_low = CisdMeasCfg(
        memory_mode="low",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    ctx_high = build_meas_ctx(ham, trial, cfg_high)
    ctx_low = build_meas_ctx(ham, trial, cfg_low)

    for i in range(1):
        walker = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), nocc_full + nvir_full, nocc_full, mix=0.25
        )
        e_high = energy_kernel_rw_rh(walker, ham, ctx_high, trial)
        e_low = energy_kernel_rw_rh(walker, ham, ctx_low, trial)
        assert jnp.allclose(e_low, e_high, rtol=1e-12, atol=1e-12), (e_low, e_high)


@pytest.mark.parametrize("norb,nocc,n_chol,memory_mode", [(8, 3, 10, "low"), (10, 4, 12, "high")])
def test_auto_force_bias_matches_manual_cisd(norb, nocc, n_chol, memory_mode):
    walker_kind = "restricted"
    key = jax.random.PRNGKey(123)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_cisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=lambda sys: make_cisd_trial_ops(sys, memory_mode=memory_mode),
        make_meas_ops_fn=lambda sys: make_cisd_meas_ops(sys, memory_mode=memory_mode),
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(1):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        # CISD overlap is more structured than RHF; auto (finite-diff / overlap-derivative)
        # can need a slightly looser tolerance.
        assert jnp.allclose(v_a, v_m, rtol=2e-5, atol=2e-6), (v_a, v_m)


@pytest.mark.parametrize("norb,nocc,n_chol,memory_mode", [(8, 3, 10, "low"), (10, 4, 12, "high")])
def test_auto_energy_matches_manual_cisd(norb, nocc, n_chol, memory_mode):
    walker_kind = "restricted"
    key = jax.random.PRNGKey(456)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_cisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
        ),
        make_trial_ops_fn=lambda sys: make_cisd_trial_ops(sys, memory_mode=memory_mode),
        make_meas_ops_fn=lambda sys: make_cisd_meas_ops(sys, memory_mode=memory_mode),
    )

    if not meas_manual.has_kernel(k_energy):
        pytest.skip("manual CISD meas does not provide k_energy")

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(1):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=2e-5, atol=2e-6), (ea, em)


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -75.72869718476204, 0.0002352938315467452),
    ],
)
def test_calc_rhf_hamiltonian(mycc, params, walker_kind, e_ref, err_ref):
    myafqmc = Afqmc(mycc)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()

    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


def test_stage_infers_trial_freeze_from_list_valued_cc_frozen(mycc_frozen_list):
    staged = Afqmc(mycc_frozen_list).stage()

    assert staged.ham.frozen == 0
    assert staged.ham.norb == mycc_frozen_list._scf.mo_coeff.shape[-1]
    assert staged.trial.kind == "cisd"
    assert staged.trial.data["ci1"].shape == mycc_frozen_list.t1.shape
    assert int(staged.trial.data["nocc_t_core"].item()) == 1
    assert int(staged.trial.data["nvir_t_outer"].item()) == 1


def test_stage_splits_list_valued_cc_frozen_with_afqmc_frozen_core(mycc_frozen_list):
    staged = Afqmc(mycc_frozen_list, norb_frozen_core=1).stage()

    assert staged.ham.frozen == 1
    assert staged.ham.norb == mycc_frozen_list._scf.mo_coeff.shape[-1] - 1
    assert staged.trial.kind == "cisd"
    assert staged.trial.data["ci1"].shape == mycc_frozen_list.t1.shape
    assert int(staged.trial.data["nocc_t_core"].item()) == 0
    assert int(staged.trial.data["nvir_t_outer"].item()) == 1


def test_stage_legacy_norb_frozen_alias_still_works(mycc_frozen_list):
    staged = Afqmc(mycc_frozen_list, norb_frozen=1).stage()

    assert staged.ham.frozen == 1
    assert staged.ham.norb == mycc_frozen_list._scf.mo_coeff.shape[-1] - 1


def test_stage_infers_integer_cc_frozen_for_hamiltonian_and_trial(mycc_frozen_int):
    staged = Afqmc(mycc_frozen_int).stage()

    assert staged.ham.frozen == 1
    assert staged.ham.norb == mycc_frozen_int._scf.mo_coeff.shape[-1] - 1
    assert staged.trial.kind == "cisd"
    assert staged.trial.frozen == 1
    assert staged.trial.data["ci1"].shape == mycc_frozen_int.t1.shape


def test_freeze_core_from_mo_cholesky_matches_pyscf_get_h1eff():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    norb_frozen = 1
    mo_coeff = np.asarray(mf.mo_coeff)
    nmo = int(mo_coeff.shape[1])
    ncas = nmo - norb_frozen
    nelecas = mol.nelectron - 2 * norb_frozen

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.mo_coeff = mo_coeff  # type: ignore
    h1_ref, ecore_ref = mc.get_h1eff()  # type: ignore

    hcore = np.asarray(mf.get_hcore())
    h1 = mo_coeff.T.conj() @ hcore @ mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), nmo)
    chol = _exact_cholesky_from_mo_eri(eri)

    ecore, h1_eff, chol_act, nelec = _freeze_core_from_mo_cholesky(
        h0=float(mf.energy_nuc()),
        h1=np.asarray(h1),
        chol=chol,
        norb_frozen=norb_frozen,
        nelec=tuple(int(x) for x in mol.nelec),  # type: ignore
    )

    assert np.allclose(h1_eff, h1_ref, atol=1e-9)
    assert np.isclose(ecore, ecore_ref, atol=1e-9)
    assert chol_act.shape[1:] == h1_ref.shape
    assert nelec == tuple(int(x) for x in mc.nelecas)  # type: ignore


def test_stage_prefers_cc_mo_coeff_for_hamiltonian(mycc_rotated_basis):
    staged = Afqmc(mycc_rotated_basis).stage()

    c_cc = np.asarray(mycc_rotated_basis.mo_coeff)
    c_scf = np.asarray(mycc_rotated_basis._scf.mo_coeff)
    hcore = np.asarray(mycc_rotated_basis._scf.get_hcore())

    h1_cc = c_cc.T.conj() @ hcore @ c_cc
    h1_scf = c_scf.T.conj() @ hcore @ c_scf

    assert np.allclose(staged.ham.h1, h1_cc)
    assert not np.allclose(h1_cc, h1_scf)


def test_cisd_host_setup_builders_match_default_builders(mycc):
    staged = Afqmc(mycc).stage()
    ham_data = _make_ham_data(staged.ham, None, compact_chol=False)
    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="restricted")
    trial_data = make_cisd_trial_data(staged.trial.data, sys)
    trial_rdm1 = make_cisd_trial_ops(sys).get_rdm1(trial_data)

    meas_ops = make_cisd_meas_ops(sys, mixed_precision=True)
    cfg = get_cisd_meas_cfg(meas_ops)
    assert cfg is not None
    assert isinstance(cfg, CisdMeasCfg)

    prop_host = _build_restricted_prop_ctx_from_host(
        staged,
        trial_rdm1=trial_rdm1,
        dt=0.005,
        mixed_precision=True,
        mesh=None,
    )
    prop_ref = _build_prop_ctx(ham_data, trial_rdm1, 0.005, chol_flat_precision=jnp.float32)

    assert jnp.allclose(prop_host.dt, prop_ref.dt)
    assert jnp.allclose(prop_host.sqrt_dt, prop_ref.sqrt_dt)
    assert jnp.allclose(prop_host.exp_h1_half, prop_ref.exp_h1_half)
    assert jnp.allclose(prop_host.mf_shifts, prop_ref.mf_shifts)
    assert jnp.allclose(prop_host.h0_prop, prop_ref.h0_prop)
    assert jnp.allclose(prop_host.chol_flat, prop_ref.chol_flat)
    assert prop_host.norb == prop_ref.norb

    ctx_host = _build_cisd_meas_ctx_from_host(staged, trial_data, cfg=cfg, mesh=None)
    ctx_ref = build_meas_ctx(ham_data, trial_data, cfg=cfg)

    assert jnp.allclose(ctx_host.rot_chol, ctx_ref.rot_chol)
    assert jnp.allclose(ctx_host.lci1, ctx_ref.lci1)
    assert ctx_host.cfg == ctx_ref.cfg


def test_setup_uses_cisd_host_runtime_layout(mycc):
    job = setup_job(mycc)
    assert isinstance(job.runtime_layout, CisdHostRuntimeLayout)


@pytest.fixture(scope="module")
def mycc():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_rotated_basis():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    nocc = mol.nelectron // 2
    nvir = mf.mo_coeff.shape[-1] - nocc

    rot_occ = np.eye(nocc)
    theta_occ = 0.31
    rot_occ[:2, :2] = np.array(
        [
            [np.cos(theta_occ), -np.sin(theta_occ)],
            [np.sin(theta_occ), np.cos(theta_occ)],
        ]
    )

    rot_vir = np.eye(nvir)
    theta_vir = -0.47
    rot_vir[:2, :2] = np.array(
        [
            [np.cos(theta_vir), -np.sin(theta_vir)],
            [np.sin(theta_vir), np.cos(theta_vir)],
        ]
    )

    mo_coeff = np.array(mf.mo_coeff, copy=True)
    mo_coeff[:, :nocc] = mo_coeff[:, :nocc] @ rot_occ
    mo_coeff[:, nocc:] = mo_coeff[:, nocc:] @ rot_vir

    mycc = cc.CCSD(mf, mo_coeff=mo_coeff)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_frozen_int():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf, frozen=1)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_frozen_list():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    frozen = [0, int(mf.mo_coeff.shape[-1] - 1)]
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_frozen_ndarray():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    frozen = np.array([0, int(mf.mo_coeff.shape[-1] - 1)], dtype=np.int64)
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
