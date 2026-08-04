from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import pytest
from pyscf import gto, scf

from trot import testing
from trot.core.ops import k_energy, k_force_bias
from trot.meas.ghf import make_ghf_meas_ops_chol
from trot.prop.blocks import block
from trot.prop.types import QmcParams
from trot.trial.ghf import GhfTrial, make_ghf_trial_ops


def _make_ghf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> GhfTrial:
    ne = nup + ndn
    mo = testing.rand_orthonormal_cols(key, 2 * norb, ne, dtype=dtype)
    return GhfTrial(mo_coeff=mo)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_ghf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
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
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ghf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ghf_trial_ops,
        make_meas_ops_fn=make_ghf_meas_ops_chol,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        # AUTO uses overlap-derivatives; for GHF you may need slightly looser tol than RHF.
        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_ghf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(1)
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
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ghf_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ghf_trial_ops,
        make_meas_ops_fn=make_ghf_meas_ops_chol,
    )

    # Some implementations may not define energy for some walker kinds; skip in that case.
    if not meas_manual.has_kernel(k_energy):
        pytest.skip(f"manual GHF meas does not provide '{k_energy}' for walker_kind={walker_kind}")

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)


def _prep(mf, walker_kind):
    (
        sys,
        ham_data,
        trial_ops,
        prop_ops,
        meas_ops,
    ) = testing.make_common_pyscf(
        mf,
        make_ghf_meas_ops_chol,
        make_ghf_trial_ops,
        walker_kind,
        ham_basis="generalized",
    )
    import numpy as np

    c = mf.mo_coeff
    overlap = mf.get_ovlp(mf.mol)
    q, r = np.linalg.qr(c.T @ overlap @ c)
    sgn = np.sign(r.diagonal())
    mo = jnp.einsum("ij,j->ij", q, sgn)
    mo = mo[:, : sys.nup + sys.ndn]
    trial_data = GhfTrial(mo)

    return sys, ham_data, trial_data, trial_ops, prop_ops, meas_ops


def mf():
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
    )
    mf = scf.GHF(mol).newton().x2c()  # type: ignore
    mf.kernel()
    return mf


mf = mf()


@pytest.mark.parametrize(
    "mf, walker_kind, e_ref, err_ref",
    [
        (mf, "generalized", -55.45498682945663, 0.007636218322027936),
    ],
)
def test_calc_ghf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    (
        sys,
        ham_data,
        trial_data,
        trial_ops,
        prop_ops,
        meas_ops,
    ) = _prep(mf, walker_kind)

    block_fn = block

    mean, err, block_e_all, block_w_all = testing.run_calc(
        sys,
        meas_ops,
        ham_data,
        trial_ops,
        trial_data,
        params,
        block_fn,
        prop_ops,
    )
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
        error_method="blocking",
    )


if __name__ == "__main__":
    pytest.main([__file__])
