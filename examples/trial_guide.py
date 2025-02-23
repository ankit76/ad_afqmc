import numpy as np
from ad_afqmc import driver, hamiltonian, propagation, sampling, wavefunctions, mpi_jax, stat_utils
from jax import numpy as jnp
import h5py
import pickle
from jax import jit, jvp, lax, vjp, vmap

def extract(a, string1, string2, accu):
    data = a[string1].item()
    tmp = [elem for elem in data[string2]]
    for elem in tmp:
        accu.append(elem)


ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
    mpi_jax._prep_afqmc()
)


a = np.load("block.npz", allow_pickle=True)
walkers = a["walkers"]
weights = a["weights"]


norb = trial.norb
nelec_sp = trial.nelec
#nelec_sp = a["nelec_sp"]

walkers = jnp.asarray(walkers)
wts = jnp.asarray(weights)
nwalkers = walkers.shape[0]

ovlp = trial.calc_overlap(walkers, wave_data)
wts = wts / ovlp

##make HF trial
hf = np.zeros((norb, nelec_sp[0]))
hf[:nelec_sp[0], :nelec_sp[0]] = np.eye(nelec_sp[0])
hf = jnp.asarray(hf)
trial = wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
wave_data["mo_coeff"] = hf[:, : nelec_sp[0]]

ham_data = trial._build_measurement_intermediates(ham_data, wave_data)

num = trial.calc_energy(walkers, ham_data, wave_data).real
den = trial.calc_overlap(walkers, wave_data)

num = num*den*wts
den = den*wts

numArr = jnp.asarray(num)
denArr = jnp.asarray(den)
nd = jnp.mean(num*den)
d2 = jnp.mean(den*den)
n2 = jnp.mean(num*num)
n = jnp.mean(num)
d = jnp.mean(den)
N = 1.*numArr.shape[0]


Energy = n/d + (- (nd - n*d)/d**2 + n/d**3 * (d2 - d**2) )/(N-1.)
var = (1./d**2 * (n2 - n**2) + n**2/d**4 * (d2 - d**2) - 2*n/d**3 * (nd - n*d) )/(N-1.)

print(Energy.real, var.real**0.5)

