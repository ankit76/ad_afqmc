import numpy as np

from ad_afqmc import config
config.setup_jax()
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random
from ad_afqmc import hamiltonian, propagation, wavefunctions
from ad_afqmc import propagation_v2, wavefunctions_v2

# -----------------------------------------------------------------------------
# Fixed Hamiltonian objects.
seed = 102
np.random.seed(seed)
n_walkers, norb, nelec, nchol = 10, 10, (5, 5), 5
h0 = np.random.rand(1)[0]
h1 = jnp.array(np.random.rand(2, norb, norb))
chol = jnp.array(np.random.rand(nchol, norb * norb))
ham_handler = hamiltonian.hamiltonian(norb)

# -----------------------------------------------------------------------------
# RHF propagator.
trial = wavefunctions_v2.rhf(norb, nelec)
prop_handler = propagation_v2.propagator_restricted(n_walkers=n_walkers, n_batch=5)

wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)

ham_data = {}
ham_data["h0"] = h0
ham_data["h1"] = h1[0]
ham_data["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data["chol"] = chol
ham_data["ene0"] = 0.0
ham_data = ham_handler.build_propagation_intermediates(
    ham_data, prop_handler, trial, wave_data
)
ham_data = ham_handler.build_measurement_intermediates(
        ham_data, trial, wave_data
)

prop_data = prop_handler.init_prop_data(trial, wave_data, ham_data)
prop_data["key"] = random.PRNGKey(seed)
prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)

fields = random.normal(
    random.PRNGKey(seed), shape=(prop_handler.n_walkers, ham_data["chol"].shape[0])
)

# -----------------------------------------------------------------------------
# UHF propagator.
nelec_sp = (5, 4)
trial_u = wavefunctions_v2.uhf(norb, nelec_sp)
prop_handler_u = propagation_v2.propagator_unrestricted(n_walkers=n_walkers, n_batch=5)

wave_data_u = {}
wave_data_u["mo_coeff"] = [
    jnp.array(np.random.rand(norb, nelec_sp[0])),
    jnp.array(np.random.rand(norb, nelec_sp[1])),
]
wave_data_u["rdm1"] = jnp.array(
    [
        jnp.array(wave_data_u["mo_coeff"][0] @ wave_data_u["mo_coeff"][0].T),
        jnp.array(wave_data_u["mo_coeff"][1] @ wave_data_u["mo_coeff"][1].T),
    ]
)

ham_data_u = {}
ham_data_u["h0"] = h0
ham_data_u["h1"] = h1
ham_data_u["chol"] = chol
ham_data_u["ene0"] = 0.0
ham_data_u = ham_handler.build_propagation_intermediates(
    ham_data_u, prop_handler_u, trial_u, wave_data_u
)
ham_data_u = ham_handler.build_measurement_intermediates(
    ham_data_u, trial_u, wave_data_u
)

prop_data_u = prop_handler_u.init_prop_data(trial_u, wave_data_u, ham_data_u)
prop_data_u["key"] = random.PRNGKey(seed)
prop_data_u["overlaps"] = trial_u.calc_overlap(prop_data_u["walkers"], wave_data_u)

# -----------------------------------------------------------------------------
# GHF propagator from UHF.
nocc = sum(nelec_sp)
trial_g = wavefunctions_v2.ghf(norb, nelec_sp)
prop_handler_g = propagation_v2.propagator_generalized(n_walkers=n_walkers, n_batch=5)

wave_data_g = {}
wave_data_g["mo_coeff"] = jsp.linalg.block_diag(*wave_data_u["mo_coeff"])
wave_data_g["rdm1"] = wave_data_g["mo_coeff"] @ wave_data_g["mo_coeff"].T

ham_data_g = {}
ham_data_g["h0"] = h0
ham_data_g["h1"] = h1
ham_data_g["chol"] = chol
ham_data_g["ene0"] = 0.0
ham_data_g = ham_handler.build_propagation_intermediates(
    ham_data_g, prop_handler_g, trial_g, wave_data_g
)
ham_data_g = ham_handler.build_measurement_intermediates(
    ham_data_g, trial_g, wave_data_g
)

init_walkers = np.zeros((n_walkers, 2*norb, nocc), dtype=np.complex128)
for iw in range(n_walkers):
    init_walkers[iw] = jsp.linalg.block_diag(
        prop_data_u["walkers"][0][iw, :, :nelec_sp[0]],
        prop_data_u["walkers"][1][iw, :, :nelec_sp[1]],
    )
init_walkers = jnp.array(init_walkers)

prop_data_g = prop_handler_g.init_prop_data(
    trial_g, wave_data_g, ham_data_g, init_walkers)
prop_data_g["key"] = prop_data_u["key"]
prop_data_g["overlaps"] = trial_g.calc_overlap(prop_data_g["walkers"], wave_data_g)

# -----------------------------------------------------------------------------
# UHF-CPMC propagator.
trial_cpmc_u = wavefunctions_v2.uhf_cpmc(norb, nelec_sp)
prop_handler_cpmc_u = propagation_v2.propagator_cpmc(n_walkers=n_walkers)

ham_data_cpmc_u = {}
ham_data_cpmc_u["h0"] = h0
ham_data_cpmc_u["h1"] = h1
ham_data_cpmc_u["chol"] = chol
ham_data_cpmc_u["ene0"] = 0.0
ham_data_cpmc_u["u"] = 4.0
ham_data_cpmc_u = ham_handler.build_propagation_intermediates(
    ham_data_cpmc_u, prop_handler_cpmc_u, trial_cpmc_u, wave_data_u
)
ham_data_cpmc_u = ham_handler.build_measurement_intermediates(
    ham_data_cpmc_u, trial_cpmc_u, wave_data_u
)

prop_data_cpmc_u = prop_handler_cpmc_u.init_prop_data(
    trial_cpmc_u, wave_data_u, ham_data_cpmc_u
)
prop_data_cpmc_u["key"] = random.PRNGKey(seed)
prop_data_cpmc_u["overlaps"] = trial_cpmc_u.calc_overlap(
    prop_data_cpmc_u["walkers"], wave_data_u
)

# -----------------------------------------------------------------------------
# GHF-CPMC propagator.
trial_cpmc_g = wavefunctions_v2.ghf_cpmc(norb, nelec_sp)
prop_handler_cpmc_g = propagation_v2.propagator_cpmc_generalized(n_walkers=n_walkers)

ham_data_cpmc_g = {}
ham_data_cpmc_g["h0"] = h0
ham_data_cpmc_g["h1"] = h1
ham_data_cpmc_g["chol"] = chol
ham_data_cpmc_g["ene0"] = 0.0
ham_data_cpmc_g["u"] = 4.0
ham_data_cpmc_g = ham_handler.build_propagation_intermediates(
    ham_data_cpmc_g, prop_handler_cpmc_g, trial_cpmc_g, wave_data_g
)
ham_data_cpmc_g = ham_handler.build_measurement_intermediates(
    ham_data_cpmc_g, trial_cpmc_g, wave_data_g
)

prop_data_cpmc_g = prop_handler_cpmc_g.init_prop_data(
    trial_cpmc_g, wave_data_g, ham_data_cpmc_g, init_walkers
)
prop_data_cpmc_g["key"] = prop_data_cpmc_u["key"]
prop_data_cpmc_g["overlaps"] = trial_cpmc_g.calc_overlap(
    prop_data_cpmc_g["walkers"], wave_data_g
)

# -----------------------------------------------------------------------------
# UHF-CPMC-nn propagator.
neighbors = tuple((i, (i + 1) % norb) for i in range(norb))
prop_handler_cpmc_nn_u = propagation_v2.propagator_cpmc_nn(n_walkers=n_walkers, neighbors=neighbors)

ham_data_cpmc_nn_u = {}
ham_data_cpmc_nn_u["h0"] = h0
ham_data_cpmc_nn_u["h1"] = h1
ham_data_cpmc_nn_u["chol"] = chol
ham_data_cpmc_nn_u["ene0"] = 0.0
ham_data_cpmc_nn_u["u"] = 4.0
ham_data_cpmc_nn_u["u_1"] = 1.0
ham_data_cpmc_nn_u = ham_handler.build_propagation_intermediates(
    ham_data_cpmc_nn_u, prop_handler_cpmc_nn_u, trial_cpmc_u, wave_data_u
)
ham_data_cpmc_nn_u = ham_handler.build_measurement_intermediates(
    ham_data_cpmc_nn_u, trial_cpmc_u, wave_data_u
)

prop_data_cpmc_nn_u = prop_handler_cpmc_nn_u.init_prop_data(
    trial_cpmc_u, wave_data_u, ham_data_cpmc_nn_u
)
prop_data_cpmc_nn_u["key"] = random.PRNGKey(seed)
prop_data_cpmc_nn_u["overlaps"] = trial_cpmc_u.calc_overlap(
    prop_data_cpmc_nn_u["walkers"], wave_data_u
)

# -----------------------------------------------------------------------------
# GHF tests.
def test_apply_trotprop_g():
    walkers_u = prop_data_u["walkers"]
    walkers_g = np.zeros((n_walkers, 2*norb, nocc), dtype=np.complex128)
    walkers_g[:, : norb, : nelec_sp[0]] = walkers_u[0]
    walkers_g[:, norb :, nelec_sp[0] :] = walkers_u[1]
    
    walkers_new = prop_handler_g._apply_trotprop(
        ham_data_g, walkers_g, fields
    )
    walkers_new_ref = prop_handler_u._apply_trotprop(
        ham_data_u, walkers_u, fields
    )

    for iw in range(n_walkers):
        np.testing.assert_allclose(
            walkers_new[iw], 
            jsp.linalg.block_diag(walkers_new_ref[0][iw], walkers_new_ref[1][iw])
        )

def test_propagate_g():
    prop_data_new = prop_handler_g.propagate(
        trial_g, ham_data_g, prop_data_g, fields, wave_data_g
    )
    prop_data_new_ref = prop_handler_u.propagate(
        trial_u, ham_data_u, prop_data_u, fields, wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"], 
        prop_data_new_ref["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"], 
        prop_data_new_ref["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][:, : norb, : nelec_sp[0]], 
        prop_data_new_ref["walkers"][0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][:, norb :, nelec_sp[0] :], 
        prop_data_new_ref["walkers"][1]
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_ref["pop_control_ene_shift"]
    )


# -----------------------------------------------------------------------------
# UHF-CPMC tests.
def test_propagate_cpmc_u():
    prop_handler_cpmc_slow = propagation_v2.propagator_cpmc_slow(n_walkers=10)
    prop_data_new = prop_handler_cpmc_u.propagate(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, fields, wave_data_u
    )
    prop_data_new_slow = prop_handler_cpmc_slow.propagate(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, fields, wave_data_u
    )
    greens_new_slow = trial_cpmc_u.calc_green_full(
        prop_data_new_slow["walkers"], wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"], 
        prop_data_new_slow["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"], 
        prop_data_new_slow["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][0], 
        prop_data_new_slow["walkers"][0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][1], 
        prop_data_new_slow["walkers"][1]
    )
    np.testing.assert_allclose(
        prop_data_new["greens"], 
        greens_new_slow
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_slow["pop_control_ene_shift"]
    )

# -----------------------------------------------------------------------------
# GHF-CPMC tests.
def test_apply_trotprop_cpmc_g():
    walkers_u = prop_data_cpmc_u["walkers"]
    walkers_g = np.zeros((n_walkers, 2*norb, nocc), dtype=np.complex128)
    walkers_g[:, : norb, : nelec_sp[0]] = walkers_u[0]
    walkers_g[:, norb :, nelec_sp[0] :] = walkers_u[1]
    
    walkers_new = prop_handler_cpmc_g._apply_trotprop(
        ham_data_cpmc_g, walkers_g, fields
    )
    walkers_new_ref = prop_handler_cpmc_u._apply_trotprop(
        ham_data_cpmc_u, walkers_u, fields
    )

    for iw in range(n_walkers):
        np.testing.assert_allclose(
            walkers_new[iw], 
            jsp.linalg.block_diag(walkers_new_ref[0][iw], walkers_new_ref[1][iw])
        )

def test_propagate_cpmc_g():
    np.testing.assert_allclose(
            prop_data_cpmc_g["overlaps"],
            prop_data_cpmc_u["overlaps"]
    )
    np.testing.assert_allclose(
            prop_data_cpmc_g["weights"],
            prop_data_cpmc_u["weights"]
    )

    prop_data_new = prop_handler_cpmc_g.propagate(
        trial_cpmc_g, ham_data_cpmc_g, prop_data_cpmc_g, fields, wave_data_g
    )
    prop_data_new_ref = prop_handler_cpmc_u.propagate(
        trial_cpmc_u, ham_data_cpmc_u, prop_data_cpmc_u, fields, wave_data_u
    )
    _greens_new_ref = trial_cpmc_u.calc_green_full(
        prop_data_new_ref["walkers"], wave_data_u
    )
    greens_new_ref = np.zeros((n_walkers, 2*norb, 2*norb))

    for iw in range(n_walkers):
        greens_new_ref[iw] = jsp.linalg.block_diag(
            _greens_new_ref[iw, 0], _greens_new_ref[iw, 1]
        )

    np.testing.assert_allclose(
        prop_data_new["overlaps"], 
        prop_data_new_ref["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"], 
        prop_data_new_ref["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][:, : norb, : nelec_sp[0]], 
        prop_data_new_ref["walkers"][0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][:, norb :, nelec_sp[0] :], 
        prop_data_new_ref["walkers"][1]
    )
    np.testing.assert_allclose(
        prop_data_new["greens"], 
        greens_new_ref
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_ref["pop_control_ene_shift"]
    )


def test_propagate_cpmc_nn_u():
    # TODO: test fails
    prop_handler_cpmc_nn_slow = propagation_v2.propagator_cpmc_nn_slow(
        n_walkers=n_walkers, neighbors=neighbors
    )
    prop_data_new = prop_handler_cpmc_nn_u.propagate(
        trial_cpmc_u, ham_data_cpmc_nn_u, prop_data_cpmc_nn_u, fields, wave_data_u
    )
    prop_data_new_slow = prop_handler_cpmc_nn_slow.propagate(
        trial_cpmc_u, ham_data_cpmc_nn_u, prop_data_cpmc_nn_u, fields, wave_data_u
    )
    new_greens_slow = trial_cpmc_u.calc_green_full(
        prop_data_new_slow["walkers"], wave_data_u
    )

    np.testing.assert_allclose(
        prop_data_new["overlaps"],
        prop_data_new_slow["overlaps"]
    )
    np.testing.assert_allclose(
        prop_data_new["weights"],
        prop_data_new_slow["weights"]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][0],
        prop_data_new_slow["walkers"][0]
    )
    np.testing.assert_allclose(
        prop_data_new["walkers"][1],
        prop_data_new_slow["walkers"][1]
    )
    np.testing.assert_allclose(
        prop_data_new["greens"], 
        new_greens_slow
    )
    np.testing.assert_allclose(
        prop_data_new["pop_control_ene_shift"], 
        prop_data_new_slow["pop_control_ene_shift"]
    )


if __name__ == "__main__":
    #test_apply_trotprop_g()
    #test_propagate_g()
    #test_propagate_cpmc_u()
    test_propagate_cpmc_g()
    #test_propagate_cpmc_nn_u()
