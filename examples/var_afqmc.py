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

walkerA = walkers[:nwalkers//2]
walkerB = walkers[nwalkers//2:]
wtsA = wts[:nwalkers//2]
wtsB = wts[nwalkers//2:]
ovlpA = trial.calc_overlap(walkerA, wave_data)
ovlpB = trial.calc_overlap(walkerB, wave_data)
wtsA = wtsA / ovlpA
wtsB = wtsB / ovlpB


#each sample is walkers coming from a given time slice
walkerA = walkerA.reshape(-1, options["n_walkers"], walkerA.shape[-2], walkerA.shape[-1])
walkerB = walkerB.reshape(-1, options["n_walkers"], walkerA.shape[-2], walkerA.shape[-1])
wtsA = wtsA.reshape(-1, options["n_walkers"])
wtsB = wtsB.reshape(-1, options["n_walkers"])



ham_data = trial._build_measurement_intermediates(ham_data, wave_data)


print(nwalkers)

ham_data["chol"] = ham_data["chol"].reshape(-1, norb, norb)
ham_data["h1"] = 1. * ham_data["h1"][0]

chol = ham_data["chol"]
h1 = ham_data["h1"]
ene0 = ham_data["h0"]

def _calc_overlap(walkerA, walkerB):
    return jnp.linalg.det( walkerA.conj().T @ walkerB )**2
calc_overlap = vmap(_calc_overlap, in_axes=(None,0))
calc_overlap12 = vmap(vmap(_calc_overlap, in_axes=(None,0)), in_axes=(0,None))

def _calc_gf(walkerA, walkerB):
    return walkerB @ jnp.linalg.inv( walkerA.conj().T @ walkerB ) @ walkerA.conj().T
calc_gf = vmap(_calc_gf, in_axes=(None,0))

def _calc_energy(walkerA, ham_data, walkerB):
    gf = _calc_gf(walkerA, walkerB)
    e1 = 2*jnp.einsum('ij,ij', gf, h1)

    f = jnp.einsum("gij,jk->gik", chol, gf.T, optimize="optimal")
    c = vmap(jnp.trace)(f)
    exc = jnp.sum(vmap(lambda x: x * x.T)(f))
    return 2.0 * jnp.sum(c * c) - exc + ene0 + e1
calc_energy = vmap(_calc_energy, in_axes=(None,None,0))
calc_energy12 = vmap(vmap(_calc_energy, in_axes=(None,None,0)), in_axes=(0,None,None))


ni, nj = 10,10

num, den = 0., 0.
numArr, denArr = [], []
for i in range(ni):
    for j in range(nj):
        ovlp12 = calc_overlap12(walkerA[i], walkerB[j])
        ene12 = calc_energy12(walkerA[i], ham_data, walkerB[j])
        numij = jnp.einsum('i,ij,j', wtsA[i].conj(), ene12*ovlp12, wtsB[j], optimize='optimal')
        denij = jnp.einsum('i,ij,j', wtsA[i].conj(), ovlp12, wtsB[j], optimize='optimal')
        numArr.append(numij)
        denArr.append(denij)

        num += numij
        den += denij
        print(i, j,  (numij/denij).real, (num/den).real, flush=True,)

numArr = jnp.asarray(numArr)
denArr = jnp.asarray(denArr)
nd = jnp.mean(numArr*denArr)
d2 = jnp.mean(denArr*denArr)
n2 = jnp.mean(numArr*numArr)
n = jnp.mean(numArr)
d = jnp.mean(denArr)
N = 1.*numArr.shape[0]


Energy = n/d + (- (nd - n*d)/d**2 + n/d**3 * (d2 - d**2) )/(N-1.)
var = (1./d**2 * (n2 - n**2) + n**2/d**4 * (d2 - d**2) - 2*n/d**3 * (nd - n*d) )/(N-1.)
# print("final energy", num/den, flush=True)

# jackknife = stat_utils.jackknife_ratios(numArr, denArr)
print(Energy.real, var.real**0.5)

