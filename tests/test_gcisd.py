from trot import config

config.configure_once()


import jax
import jax.numpy as jnp
import pytest
from pyscf import cc, gto, scf

from trot import testing
from trot.core.ops import k_energy, k_force_bias
from trot.meas.gcisd import make_gcisd_meas_ops
from trot.prop.blocks import block
from trot.prop.types import QmcParams
from trot.trial.gcisd import GcisdTrial, make_gcisd_trial_ops


def _make_gcisd_trial(
    key,
    norb: int,
    nup: int,
    ndn: int,
    *,
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
) -> GcisdTrial:
    """
    Random GCISD coefficients in the MO basis where the reference occupies
    [0..nocc-1].

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    norb = 2 * norb
    nocc = nup + ndn
    nvir = norb - nocc
    k1, k2 = jax.random.split(key)

    c1 = scale_ci1 * jax.random.normal(k1, (nocc, nvir), dtype=dtype)
    c2 = scale_ci2 * jax.random.normal(k2, (nocc, nvir, nocc, nvir), dtype=dtype)

    # Antisymmetry
    c2 = 0.25 * (
        c2
        - jnp.einsum("iajb->jaib", c2)
        - jnp.einsum("iajb->ibja", c2)
        + jnp.einsum("iajb->jbia", c2)
    )

    c = jnp.eye(norb, norb)

    return GcisdTrial(
        mo_coeff=c,
        c1=c1,
        c2=c2,
    )


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("generalized", 6, 3, 2, 12),
        ("generalized", 10, 4, 3, 12),
    ],
)
def test_auto_force_bias_matches_manual_gcisd(walker_kind, norb, nup, ndn, n_chol):
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
        make_trial_fn=_make_gcisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_gcisd_trial_ops,
        make_meas_ops_fn=make_gcisd_meas_ops,
        ham_basis="generalized",
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(1):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, atol=1e-12), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("generalized", 6, 3, 2, 12),
    ],
)
def test_auto_energy_matches_manual_gcisd(walker_kind, norb, nup, ndn, n_chol):
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
        make_trial_fn=_make_gcisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_gcisd_trial_ops,
        make_meas_ops_fn=make_gcisd_meas_ops,
        ham_basis="generalized",
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(1):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        e_m = e_manual(wi, ham, ctx_manual, trial)
        e_a = e_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(e_a, e_m, rtol=5e-6, atol=5e-7), (e_a, e_m)


def _prep(mycc, walker_kind):

    mf = mycc._scf
    (
        sys,
        ham_data,
        trial_ops,
        prop_ops,
        meas_ops,
    ) = testing.make_common_pyscf(
        mf,
        make_gcisd_meas_ops,
        make_gcisd_trial_ops,
        walker_kind,
        ham_basis="generalized",
    )
    import numpy as np

    def get_gcisd(cc):
        ci2 = (
            np.einsum("ijab->iajb", cc.t2)
            + np.einsum("ia,jb->iajb", cc.t1, cc.t1)
            - np.einsum("ib,ja->iajb", cc.t1, cc.t1)
        )

        ci1 = jnp.array(cc.t1)
        ci2 = jnp.array(ci2)

        return ci1, ci2

    ci1, ci2 = get_gcisd(mycc)
    c = mf.mo_coeff
    overlap = mf.get_ovlp(mf.mol)
    q, r = np.linalg.qr(c.T @ overlap @ c)
    sgn = np.sign(r.diagonal())
    mo = jnp.einsum("ij,j->ij", q, sgn)
    trial_data = GcisdTrial(mo_coeff=mo, c1=ci1, c2=ci2)

    return sys, ham_data, trial_data, trial_ops, prop_ops, meas_ops


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("generalized", -55.43960013399074, 0.0001081737355823328),
    ],
)
def test_calc_ghf_hamiltonian(mycc, params, walker_kind, e_ref, err_ref):
    (
        sys,
        ham_data,
        trial_data,
        trial_ops,
        prop_ops,
        meas_ops,
    ) = _prep(mycc, walker_kind)

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
def mycc():
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
    mycc = cc.GCCSD(mf)
    mycc.kernel()
    return mycc


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
