import os, time
import numpy as np
from numpy.random import Generator, MT19937, PCG64
import scipy as sp
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit, custom_jvp, vmap, random, vjp, jvp, checkpoint, dtypes
from mpi4py import MPI
import pickle
from ad_afqmc import linalg_utils, sr, sampler, propagation, stat_utils

from functools import partial
print = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def afqmc(ham_data, ham, propagator, trial, observable, options):
  init = time.time()
  nwalkers = options['n_walkers']
  seed = options['seed']
  neql = options['n_eql']

  observable_op = jnp.array(observable[0])
  observable_constant = observable[1]

  ham_data = ham.rot_ham(ham_data)
  ham_data = ham.prop_ham(ham_data, propagator.dt)

  global_block_weights = None
  global_block_energies = None
  global_block_observables = None
  if rank == 0:
    global_block_weights = np.zeros(size * propagator.n_blocks)
    global_block_energies = np.zeros(size * propagator.n_blocks)
    global_block_observables = np.zeros(size * propagator.n_blocks)
  
  prop_data = {}
  norb = ham.norb
  nelec  = ham.nelec
  prop_data['weights'] = jnp.ones(nwalkers)
  prop_data['walkers'] = jnp.stack([jnp.eye(norb, nelec) + 0.j for _ in range(nwalkers)])
  energy_samples = jnp.real(trial.calc_energy_vmap(ham_data, prop_data['walkers']))
  #global_block_energies[0] = jnp.sum(energy_samples) / nwalkers   # assuming identical walkers
  e_estimate = jnp.array(jnp.sum(energy_samples) / nwalkers)
  prop_data['e_estimate'] = e_estimate
  prop_data['pop_control_ene_shift'] = e_estimate
  total_block_energy_n = np.zeros(1, dtype='float32')
  total_block_observable_n = np.zeros(1, dtype='float32')
  total_block_weight_n = np.zeros(1, dtype='float32')
  #rng = Generator(MT19937(seed + rank))
  prop_data['key'] = random.PRNGKey(seed+rank)
  hf_rdm = 2 * np.eye(norb, nelec).dot(np.eye(norb, nelec).T)
  hf_observable = np.sum(hf_rdm * observable_op)

  comm.Barrier()
  init_time = time.time() - init
  if rank == 0:
    print("# Equilibration sweeps:")
    print("#   Iter        Block energy      Walltime")
    n = 0
    print(f"# {n:5d}      {e_estimate:.9e}     {init_time:.2e} ")
  comm.Barrier()

  propagator_eq = propagation.propagator(propagator.dt, propagator.n_steps, n_blocks=50)

  for n in range(1, neql+1):
    block_energy_n, prop_data = sampler.propagate_phaseless(ham, ham_data, propagator_eq, prop_data, trial)
    block_energy_n = np.array([block_energy_n], dtype='float32')
    block_weight_n = np.array([jnp.sum(prop_data['weights'])], dtype='float32')
    block_weighted_energy_n = np.array([block_energy_n * block_weight_n], dtype='float32')
    comm.Reduce([block_weighted_energy_n, MPI.FLOAT], [total_block_energy_n, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([block_weight_n, MPI.FLOAT], [total_block_weight_n, MPI.FLOAT], op=MPI.SUM, root=0)
    if rank == 0:
      block_weight_n = total_block_weight_n
      block_energy_n = total_block_energy_n / total_block_weight_n
    comm.Bcast(block_weight_n, root=0)
    comm.Bcast(block_energy_n, root=0)
    prop_data['walkers'] = linalg_utils.qr_vmap(prop_data['walkers'])
    prop_data['key'], subkey = random.split(prop_data['key'])
    zeta = random.uniform(subkey)
    prop_data['walkers'], prop_data['weights'] = sr.stochastic_reconfiguration_mpi(prop_data['walkers'], prop_data['weights'], zeta, comm)
    e_estimate = 0.9 * e_estimate + 0.1 * block_energy_n

    comm.Barrier()
    if rank == 0:
      print(
          f"# {n:5d}      {block_energy_n[0]:.9e}     {time.time() - init:.2e} ", flush=True)
    comm.Barrier()

  local_large_deviations = np.array(0)

  comm.Barrier()
  init_time = time.time() - init
  if rank == 0:
    print("#\n# Sampling sweeps:")
    print("#  Iter        Mean energy          Stochastic error       Mean observable       Walltime")
  comm.Barrier()

  propagate_phaseless_wrapper = lambda x, y, z: sampler.propagate_phaseless_ad(ham, ham_data, x, y, propagator, z, trial)
  prop_data_tangent = {}
  for x in prop_data:
    if prop_data[x].dtype == 'uint32':
      prop_data_tangent[x] = np.zeros(prop_data[x].shape, dtype=dtypes.float0)
    else:
      prop_data_tangent[x] = np.zeros_like(prop_data[x])

  for n in range(propagator.n_ad_blocks):
    coupling = 0.
    block_energy_n, block_observable_n, prop_data = jvp(propagate_phaseless_wrapper, (coupling, observable_op, prop_data), (1., 0. * observable_op, prop_data_tangent), has_aux=True)
    if np.isnan(block_observable_n):
      block_observable_n = hf_observable
      local_large_deviations += 1

    block_energy_n = np.array([block_energy_n], dtype='float32')
    block_observable_n = np.array([block_observable_n + observable_constant], dtype='float32')
    block_weight_n = np.array([jnp.sum(prop_data['weights'])], dtype='float32')
    gather_weights = None
    gather_energies = None
    gather_observables = None
    if rank == 0:
      gather_weights = np.zeros(size, dtype='float32')
      gather_energies = np.zeros(size, dtype='float32')
      gather_observables = np.zeros(size, dtype='float32')

    comm.Gather(block_weight_n, gather_weights, root=0)
    comm.Gather(block_energy_n, gather_energies, root=0)
    comm.Gather(block_observable_n, gather_observables, root=0)
    block_energy_n = 0.
    if rank == 0:
      global_block_weights[n * size: (n+1) * size] = gather_weights
      global_block_energies[n * size: (n+1) * size] = gather_energies
      global_block_observables[n * size: (n+1) * size] = gather_observables
      block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(gather_weights)

    block_energy_n = comm.bcast(block_energy_n, root=0)
    prop_data['walkers'] = linalg_utils.qr_vmap(prop_data['walkers'])
    prop_data['key'], subkey = random.split(prop_data['key'])
    zeta = random.uniform(subkey)
    prop_data['walkers'], prop_data['weights'] = sr.stochastic_reconfiguration_mpi(prop_data['walkers'], prop_data['weights'], zeta, comm)
    e_estimate = 0.9 * e_estimate + 0.1 * block_energy_n

    if n % (max(propagator.n_blocks//10, 1)) == 0:
      comm.Barrier()
      if rank == 0:
        e_afqmc, energy_error = stat_utils.blocking_analysis(global_block_weights[:(n+1) * size], global_block_energies[:(n+1) * size], neql=0)
        obs_afqmc, obs_error = stat_utils.blocking_analysis(global_block_weights[:(n+1) * size], global_block_observables[:(n + 1) * size], neql=0)
        if energy_error is not None:
          print(f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {obs_afqmc:.9e}       {time.time() - init:.2e} ", flush=True)
        else:
          print(f" {n:5d}      {e_afqmc:.9e}                -              {obs_afqmc:.9e}       {time.time() - init:.2e} ", flush=True)
        np.savetxt('samples_jax.dat', np.stack((global_block_weights[:(n+1) * size], global_block_energies[:(n+1) * size], global_block_observables[:(n+1) * size])).T)
      comm.Barrier()

  global_large_deviations = np.array(0)
  comm.Reduce([local_large_deviations, MPI.INT], [global_large_deviations, MPI.INT], op=MPI.SUM, root=0)
  comm.Barrier()
  if rank == 0:
    print(f'#\n# Number of large deviations: {global_large_deviations}', flush=True)

  comm.Barrier()
  if rank == 0:
    np.savetxt('samples_raw.dat', np.stack((global_block_weights, global_block_energies, global_block_observables)).T)
    samples_clean = stat_utils.reject_outliers(np.stack((global_block_weights, global_block_energies, global_block_observables)).T, 2)
    print(f'# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} ')
    np.savetxt('samples.dat', samples_clean)
    global_block_weights = samples_clean[:, 0]
    global_block_energies = samples_clean[:, 1]
    global_block_observables = samples_clean[:, 2]

    e_afqmc, err_afqmc = stat_utils.blocking_analysis(global_block_weights, global_block_energies, neql=0, printQ=True)
    if err_afqmc is not None:
      sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
      sig_err = np.around(np.round(err_afqmc * 10**sig_dec) * 10**(-sig_dec), sig_dec)
      sig_e = np.around(e_afqmc, sig_dec)
      print(f'AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n')
    elif e_afqmc is not None:
      print(f'AFQMC energy: {e_afqmc}\n', flush=True)

    obs_afqmc, err_afqmc = stat_utils.blocking_analysis(global_block_weights, global_block_observables, neql=0, printQ=True)
    if err_afqmc is not None:
      sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
      sig_err = np.around(np.round(err_afqmc * 10**sig_dec) * 10**(-sig_dec), sig_dec)
      sig_obs = np.around(obs_afqmc, sig_dec)
      print(f'AFQMC observable: {sig_obs:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n')
    elif obs_afqmc is not None:
      print(f'AFQMC observable: {obs_afqmc}\n', flush=True)
  comm.Barrier()

def run_afqmc(options=None, script=None, mpi_prefix=None, nproc=None):
  if options is None:
    options = {}
  with open('options.bin', 'wb') as f:
    pickle.dump(options, f)
  if script is None:
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)  
    script = f'{dir_path}/mpi_jax.py'
  if mpi_prefix is None:
    mpi_prefix = "mpirun "
    if nproc is not None:
      mpi_prefix += f"-np {nproc} "
  os.system(f'export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script}')


  