import numpy as np
from ad_afqmc import driver, hamiltonian, propagation, sampling, wavefunctions, mpi_jax, stat_utils
from jax import numpy as jnp
import h5py
import pickle
from jax import jit, jvp, lax, vjp, vmap, random

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


ham_data = trial._build_measurement_intermediates(ham_data, wave_data)

options["dt"] = 0.01
dt = options["dt"]
prop = propagation.propagator_restricted(
    options["dt"], options["n_walkers"], n_batch=options["n_batch"],
    phaseless_epsilon = options["phaseless_epsilon"]
)
ham_data = prop._build_propagation_intermediates(ham_data, trial, wave_data)

walk = walkers[3000:3001]
fields = random.normal(random.PRNGKey(0), shape=(1,ham_data["chol"].shape[0]))
force_bias = trial.calc_force_bias(walk, ham_data, wave_data)
field_shifts = -jnp.sqrt(dt) * (1.0j * force_bias - ham_data["mf_shifts"])
shifted_fields = fields - field_shifts
shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
fb_term = jnp.sum(
    fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
)

walkout = prop._apply_trotprop(
    ham_data, walk, shifted_fields
)
overlap = trial.calc_overlap(walk, wave_data)

overlaps_new = trial.calc_overlap(walkout, wave_data)

loc_energy = trial.calc_energy(walk, ham_data, wave_data)
imp_fun2 = jnp.exp(-prop.dt * loc_energy )
# theta = jnp.angle(jnp.exp(-self.dt * loc_energy))
# imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)

imp_fun = (
    jnp.exp(
        -jnp.sqrt(prop.dt) * shift_term
        + fb_term
        + prop.dt * (ham_data["h0_prop"])
    )
    * overlaps_new
    / overlap
)
theta = jnp.angle(
    jnp.exp(-jnp.sqrt(prop.dt) * shift_term)
    * overlaps_new
    / overlap
)
imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
print(imp_fun2, jnp.exp(fb_term), imp_fun)



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

