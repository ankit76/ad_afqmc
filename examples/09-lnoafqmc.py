import sys
import os
import numpy as np
from pyscf import gto, scf, mp, cc, lo
from pyscf.data.elements import chemcore
from pyscf.lib import logger
from ad_afqmc import lnoafqmc, lnoutils
log = logger.Logger(sys.stdout, 6)

# Assumes lno code is obtained from pyscf_forge repository: https://github.com/hongzhouye/pyscf-forge/tree/lnocc

# S22-2: water dimer
atom = """
O   -1.485163346097   -0.114724564047    0.000000000000
H   -1.868415346097    0.762298435953    0.000000000000
H   -0.533833346097    0.040507435953    0.000000000000
O    1.416468653903    0.111264435953    0.000000000000
H    1.746241653903   -0.373945564047   -0.758561000000
H    1.746241653903   -0.373945564047    0.758561000000
"""
basis = "cc-pvdz"

mol = gto.M(atom=atom, basis=basis)
mol.verbose = 4
frozen = chemcore(mol)

mf = scf.RHF(mol).density_fit()
mf.kernel()

# canonical
mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()
efull_mp2 = mmp.e_corr

lo_coeff, frag_lolist = lnoutils.prep_local_orbitals(mf, frozen=frozen)

# LNO-AFQMC calculation: here we can scan over a list of thresholds
mcc = lnoafqmc.LNOAFQMC(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
mcc.n_blocks = 50
mcc.n_walkers = 20
#mcc.n_ene_blocks=50
mcc.nproc = 2
mcc.maxError = 1e-3/np.sqrt(len(frag_lolist))

gamma = 10  # thresh_occ / thresh_vir
threshs = np.asarray([1e-4])
elno_afqmc_uncorr = np.zeros_like(threshs)
stocherr_afqmc = np.zeros_like(threshs)
elno_mp2 = np.zeros_like(threshs)

for i, thresh in enumerate(threshs):
    mcc.lno_thresh = [thresh * gamma, thresh]
    mcc.kernel()
    stocherr_afqmc[i] = mcc.afqmc_error_ecorr
    elno_afqmc_uncorr[i] = mcc.e_corr_afqmc
    elno_mp2[i] = mcc.e_corr_pt2
elno_afqmc = elno_afqmc_uncorr - elno_mp2 + efull_mp2

log.info("")
for i, thresh in enumerate(threshs):
    e0 = elno_afqmc_uncorr[i]
    err = stocherr_afqmc[i]
    e1 = elno_afqmc[i]
    log.info(
        "thresh = %.3e  E_corr(LNO-AFQMC) = %.15g +/- %.15g  E_corr(LNO-AFQMC+∆PT2) = %.15g +/- %.15g",
        thresh,
        e0,
        err,
        e1,
        err
    )


