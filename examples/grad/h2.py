from functools import partial
from functools import reduce

import numpy as np
from pyscf import fci, gto,ci, scf, grad, cc,df,lib #,ao2mo
import pyscf
from ad_afqmc import pyscf_interface, run_afqmc, linalg_utils , grad_utils
from scipy.linalg import fractional_matrix_power
import jax.numpy as jnp
print = partial(print, flush=True)
import numpy


rs = [2.4]
#rs = [2.0]
basis=  "ccpvdz" #   "631+g"

for r in rs:
  atom_symbols = np.array(["H","H"])#,"H","H"])
  coords     = np.array([[0,0,0],[0,0,r]])#,[0,0,2*r],[0,0,3*r]])

  atomstring = list(zip(atom_symbols, coords))
  mol = gto.M(atom=atomstring, basis=basis, verbose=3, unit="bohr")
  mf = df.density_fit(scf.UHF(mol))

  mf.kernel()

  options = {
      "n_eql": 10,
      "n_ene_blocks": 4,
      "n_sr_blocks": 50,
      "n_blocks": 10,
      "n_walkers": 25,
      "do_sr": True,
      "orbital_rotation": True,
      "walker_type": "uhf",
      "trial": "uhf",
      "seed": 101,
      "ad_mode": "nuc_grad",
}                          

  grad_utils.FD_integrals(mf)
  grad_utils.write_integrals_lowdins(mf)

  run_afqmc.run_afqmc(options=options,nproc=1)
  grad_utils.calculate_nuc_gradients(uhf=True)
  print(f"r = {r}")
  mf_grad = mf.Gradients()
  mf_grad.kernel()

  mc = cc.CCSD(mf)
  mc.kernel()
  g = mc.nuc_grad_method().kernel()



