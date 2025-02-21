import numpy as np
from ad_afqmc import driver, hamiltonian, propagation, sampling, wavefunctions, mpi_jax
from jax import numpy as jnp
import h5py
import pickle

def extract(a, string1, string2, accu):
    data = a[string1].item()
    tmp = [elem for elem in data[string2]]
    for elem in tmp:
        accu.append(elem)


ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    mpi_jax._prep_afqmc()
)

n_blocks = options["n_blocks"]
walkers = []
weights = []
for i in range(n_blocks):
    a = np.load("block_"+str(i)+".npz", allow_pickle=True)
    extract(a, "prop_data", "walkers", walkers) 
    extract(a, "prop_data", "weights", weights) 


norb = trial.norb
nelec_sp = trial.nelec
#nelec_sp = a["nelec_sp"]

walkers = jnp.asarray(walkers)
wts = jnp.asarray(weights)

ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
energies = trial.calc_energy(walkers, ham_data, wave_data).real

print(jnp.sum(energies * wts) / jnp.sum(wts))

