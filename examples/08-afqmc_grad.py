import numpy as np
from pyscf import gto,scf, grad, cc,df
from ad_afqmc import afqmc,grad_utils
from ad_afqmc import config
config.afqmc_config["use_gpu"] = False
config.afqmc_config["use_mpi"] = True

r = 1.05835
basis=  "sto6g" #   "631+g"

def geom(r,atom="H"):
    atomstring = f"""
    {atom} 0.0 0.0 0.0
    {atom} 0.0 0.0 {r}
    """
    return atomstring


atomstring = geom(r,atom="H")
mol = gto.M(atom=atomstring, basis=basis, verbose=3, unit="Angstrom")
mf = df.density_fit(scf.UHF(mol))
mf.kernel()

af = afqmc.AFQMC(mf)
af.nproc = 4
af.n_walkers = 50
af.n_blocks = 10
af.n_ene_blocks = 4
af.n_sr_blocks = 50
af.ad_mode = "nuc_grad"
af.tmpdir = "./"
af.kernel()

#print(f"r = {r}")
#mf_grad = mf.Gradients()
#mf_grad.kernel()
#
#mc = cc.CCSD(mf)
#mc.kernel()
#g = mc.nuc_grad_method().kernel()



