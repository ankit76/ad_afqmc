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
nwalkers = walkers.shape[0]

walkerA = walkers[:nwalkers//2]
walkerB = walkers[nwalkers//2:]
wtsA = wts[:nwalkers//2]
wtsB = wts[nwalkers//2:]
ovlpA = trial.calc_overlap(walkerA, wave_data)
ovlpB = trial.calc_overlap(walkerB, wave_data)
wtsA = wtsA / ovlpA
wtsB = wtsB / ovlpB

import pdb
pdb.set_trace()

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


ni, nj = 250, 250

num, den = 0., 0.
numArr, denArr = jnp.asarray([]), jnp.asarray([])
for i in range(ni):
    startA, stopA = i * (nwalkers//2//ni), min( (i+1) * (nwalkers//2//ni), nwalkers//2 )
    for j in range(nj):
        startB, stopB = j * (nwalkers//2//nj), min( (j+1) * (nwalkers//2//nj), nwalkers//2 )

        ovlp12 = calc_overlap12(walkerA[startA:stopA], walkerB[startB:stopB])
        ene12 = calc_energy12(walkerA[startA:stopA], ham_data, walkerB[startB:stopB])
        numij = jnp.einsum('i,ij,j', wtsA[startA:stopA].conj(), ene12*ovlp12, wtsB[startB:stopB], optimize='optimal')
        denij = jnp.einsum('i,ij,j', wtsA[startA:stopA].conj(), ovlp12, wtsB[startB:stopB], optimize='optimal')
        num += numij
        den += denij
        print( (numij/denij).real, (num/den).real)

print("final energy", num/den)


