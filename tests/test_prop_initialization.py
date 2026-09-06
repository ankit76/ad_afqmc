from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import MeasOps, TrialOps, k_energy
from trot.core.system import System
from trot.prop.afqmc import init_prop_state as init_afqmc
from trot.prop.cpmc import init_prop_state as init_cpmc
from trot.prop.types import QmcParams
from trot.trial.ccsd import make_init_prop_state as make_ccsd_initializer
from trot.trial.uccsd import make_init_prop_state as make_uccsd_initializer


@pytest.mark.parametrize("kind", ["afqmc", "cpmc", "ccsd", "uccsd"])
def test_initializers_reuse_supplied_context_and_preserve_standalone_fallback(kind):
    mo = jnp.eye(2)
    t1 = jnp.zeros((1, 1))
    t2 = jnp.zeros((1, 1, 1, 1))
    if kind == "ccsd":
        initialize = make_ccsd_initializer(mo, t1, t2)
    elif kind == "uccsd":
        initialize = make_uccsd_initializer((mo, mo), (t1, t1), (t2, t2, t2))
    else:
        initialize = init_afqmc if kind == "afqmc" else init_cpmc

    sys = System(
        norb=2, nelec=(1, 1),
        walker_kind="unrestricted" if kind == "uccsd" else "restricted",
    )
    params = QmcParams(n_walkers=2, n_chunks=1, seed=903)
    trial_data = jnp.stack([jnp.diag(jnp.array([1.0, 0.0]))] * 2)
    context = {"offset": jnp.asarray(3.5)}
    builds = []

    def build_context(ham, trial):
        builds.append((ham, trial))
        return context

    def overlap(walker, trial):
        return jnp.asarray(1.0 + 0.0j)

    def energy(walker, ham, ctx, trial):
        return ham + ctx["offset"] + sum(
            jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree_util.tree_leaves(walker)
        )

    meas_ops = MeasOps(
        overlap=overlap, build_meas_ctx=build_context, kernels={k_energy: energy}
    )
    kwargs = dict(
        sys=sys, ham_data=jnp.asarray(0.25), trial_data=trial_data,
        trial_ops=TrialOps(overlap=overlap, get_rdm1=lambda trial: trial),
        meas_ops=meas_ops, params=params,
    )
    standalone = initialize(**kwargs)
    assert len(builds) == 1
    reused = initialize(**kwargs, meas_ctx=context)
    assert len(builds) == 1
    for actual, expected in zip(
        jax.tree_util.tree_leaves(reused), jax.tree_util.tree_leaves(standalone), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    changed = initialize(**kwargs, meas_ctx={"offset": context["offset"] + 2.0})
    assert len(builds) == 1
    np.testing.assert_allclose(changed.e_estimate, standalone.e_estimate + 2.0, atol=1.0e-12)


@pytest.mark.parametrize("initialize", [init_afqmc, init_cpmc])
def test_explicit_initial_energy_does_not_build_measurement_context(initialize):
    def forbidden_context(ham, trial):
        raise AssertionError("An explicit initial energy must bypass context construction.")

    def overlap(walker, trial):
        return jnp.asarray(1.0 + 0.0j)

    state = initialize(
        sys=System(norb=2, nelec=(1, 1), walker_kind="restricted"),
        ham_data=jnp.asarray(0.25),
        trial_ops=TrialOps(overlap=overlap, get_rdm1=lambda trial: trial),
        trial_data=None,
        meas_ops=MeasOps(overlap=overlap, build_meas_ctx=forbidden_context, kernels={}),
        params=QmcParams(n_walkers=2, seed=904),
        initial_walkers=jnp.broadcast_to(jnp.eye(2, 1), (2, 2, 1)),
        initial_e_estimate=jnp.asarray(-1.5),
    )
    assert float(state.e_estimate) == -1.5
