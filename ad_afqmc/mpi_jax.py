import sys
import numpy as np
from jax import numpy as jnp
from mpi4py import MPI
import pickle
from ad_afqmc import driver, hamiltonian, propagation, wavefunctions

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import h5py
with h5py.File('FCIDUMP_chol', 'r') as fh5:
  [nelec, nmo, ms, nchol] = fh5['header']
  h0 = jnp.array(fh5.get('energy_core'))
  h1 = jnp.array(fh5.get('hcore')).reshape(nmo, nmo)
  chol = jnp.array(fh5.get('chol')).reshape(-1, nmo, nmo)

with h5py.File('observable.h5', 'r') as fh5:
  [ observable_constant ] = fh5['constant']
  observable_op = np.array(fh5.get('op')).reshape(nmo, nmo)

with open('options.bin', 'rb') as f:
  options = pickle.load(f)

norb = nmo
nelec = nelec // 2

options['dt'] = options.get('dt', 0.01)
options['n_walkers'] = options.get('n_walkers', 50)
options['n_steps'] = options.get('n_steps', 50)
options['n_sr_blocks'] = options.get('n_sr_blocks', 1)
options['n_ad_blocks'] = options.get('n_ad_blocks', 50)
options['n_blocks'] = options.get('n_blocks', 50)
options['seed'] = options.get('seed', np.random.randint(1, 1e6))
options['n_eql'] = options.get('n_eql', 1)

ham_data = {}
ham_data['h0'] = h0
ham_data['h1'] = h1
ham_data['chol'] = chol.reshape(nchol, -1)
ham = hamiltonian.hamiltonian(nmo, nelec, nchol)

prop = propagation.propagator(options['dt'], options['n_steps'], options['n_blocks'], options['n_sr_blocks'], options['n_ad_blocks'])

trial = wavefunctions.rhf(nelec)

if rank == 0:
  print(f'# norb: {norb}')
  print(f'# nelec: {2*nelec}')
  print('#')
  for op in options:
    print(f'# {op}: {options[op]}')
  print('#')

import time
init = time.time()
comm.Barrier()
driver.afqmc(ham_data, ham, prop, trial, [ observable_op, observable_constant ], options)
comm.Barrier()
end = time.time()
if rank == 0:
  print(f'ph_afqmc walltime: {end - init}', flush=True)

comm.Barrier()
