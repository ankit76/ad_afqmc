from pyscf import fci, gto, scf, cc
from pyscf.scf import hf
import numpy as np
from ad_afqmc import wavefunctions, afqmc, pyscf_interface, run_afqmc
import scipy.linalg as la
import os

tmpdir="tmp"
chol_cut = 1e-8
 
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

def transform_matrix(A):
    n = A.shape[0]
    B = np.zeros((2 * n, 2 * n), dtype=complex)

    for i in range(2 * n):
        col_idx = i // 2
        if i % 2 == 0:
            B[:n, i] = A[:, col_idx]
        else:
            B[n:, i] = A[:, col_idx]

    return B

def random_orthogonal_real(n):
    np.random.seed(42)
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q

def random_orthogonal_complex(n):
    np.random.seed(42)
    A = np.random.randn(n, n) + 1j*np.random.randn(n, n) 
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q

options = {
"dt": 0.005,
"n_eql": 2,
"n_ene_blocks": 1,
"n_sr_blocks": 5,
"n_blocks": 5,
"n_prop_steps": 50,
"n_walkers": 4,
"seed": 8,
"trial": "",
"walker_type": "",
}

def check_hf(mol, mf):
    nmo=np.shape(mf.mo_coeff)[1]
    nelec=sum(mol.nelec)
    nocc=nelec//2
    hcore_ao = hf.get_hcore(mol)
    n_ao = hcore_ao.shape[-1]


    # RHF
    pyscf_interface.prep_afqmc(mf, tmpdir=tmpdir, chol_cut=chol_cut)

    options["trial"] = "rhf"
    options["walker_type"] = "restricted"
    
    ene1, _ = run_afqmc.run_afqmc(options=options, mpi_prefix="", nproc=None, tmpdir=tmpdir)
    
    # RHF based GHF
    mo_coeff = mf.mo_coeff
    mo_coeff = transform_matrix(mo_coeff)

    pyscf_interface.prep_afqmc_ghf_complex(mol, mo_coeff+0.0j, hcore_ao, n_ao, tmpdir, chol_cut=chol_cut)

    options["trial"] = "ghf_complex"
    options["walker_type"] = "generalized"   
 
    ene2, _ = run_afqmc.run_afqmc(options=options, mpi_prefix="", nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} {ene2}"
    
    # RHF based GHF with a complex rotation
    mo_coeff = mf.mo_coeff
    mo_coeff = transform_matrix(mo_coeff)
    mo_coeff = mo_coeff @ la.block_diag(random_orthogonal_complex(nelec), random_orthogonal_complex(2*nmo-nelec))

    pyscf_interface.prep_afqmc_ghf_complex(mol, mo_coeff+0j, hcore_ao, n_ao, tmpdir, chol_cut=chol_cut)
    
    ene2, _ = run_afqmc.run_afqmc(options=options, mpi_prefix="", nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} {ene2}"

def check_cc(mol, mf):
    nmo=np.shape(mf.mo_coeff)[1]
    nelec=sum(mol.nelec)
    nocc=nelec//2
    hcore_ao = hf.get_hcore(mol)
    n_ao = hcore_ao.shape[-1]

    # RCCSD
    mycc = cc.CCSD(mf)
    mycc.kernel()

    pyscf_interface.prep_afqmc(mycc, tmpdir=tmpdir, chol_cut=chol_cut)

    options["trial"] = "cisd"
    options["walker_type"] = "restricted"

    ene1, _ = run_afqmc.run_afqmc(options=options, mpi_prefix="", nproc=None, tmpdir=tmpdir)

    # GCCSD
    ## GHF
    gmf = scf.GHF(mol)
    gmf.kernel()
    assert np.isclose(mf.e_tot, gmf.e_tot, atol=1e-8), f"{mf.e_tot} {gmf.e_tot}"

    ## GCCSD
    ccsd = cc.CCSD(gmf)
    ccsd.kernel()
    assert np.isclose(mycc.e_tot, ccsd.e_tot, atol=1e-8), f"{mycc.e_tot} vs {ccsd.e_tot}"
    np.savez(tmpdir+"/amplitudes.npz",t1=ccsd.t1, t2=ccsd.t2)

    pyscf_interface.prep_afqmc_ghf_complex(mol, gmf.mo_coeff+0j, hcore_ao, n_ao, tmpdir, chol_cut=chol_cut)

    options["trial"] = "gcisd_complex"
    options["walker_type"] = "generalized"
    
    ene2, _ = run_afqmc.run_afqmc(options=options, mpi_prefix="", nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} vs {ene2}"

# H4
mol = gto.M(atom="""
    H 0. 0. -1.2
    H 0. 0. -0.4
    H 0. 0. 0.4
    H 0. 0. 1.2
    """,
    basis="sto-3g",
    verbose=3)
mf = scf.RHF(mol)
mf.kernel()

check_hf(mol, mf)
check_cc(mol, mf)

# H2O
mol = gto.M(
    atom = f'''
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    ''',
    basis = 'sto-3g',
    spin=0,
    verbose = 3)
mf = scf.RHF(mol)
mf.kernel()

check_hf(mol, mf)
check_cc(mol, mf)
