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

def rhf_to_ghf(A):
    n = A.shape[-1]
    B = np.zeros((2 * n, 2 * n), dtype=complex)

    for i in range(2 * n):
        col_idx = i // 2
        if i % 2 == 0:
            B[:n, i] = A[:, col_idx]
        else:
            B[n:, i] = A[:, col_idx]

    return B

def uhf_to_ghf(A):
    n = A.shape[-1]
    B = np.zeros((2 * n, 2 * n), dtype=complex)

    for i in range(2 * n):
        col_idx = i // 2
        if i % 2 == 0:
            B[:n, i] = A[0][:, col_idx]
        else:
            B[n:, i] = A[1][:, col_idx]

    return B

def transform_matrix(A):
    if A.ndim == 2:
        B = rhf_to_ghf(A)
    elif A.ndim == 3:
        B = uhf_to_ghf(A)
    
    return B

def random_orthogonal_real(n):
    rng = np.random.default_rng()
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q

def random_orthogonal_complex(n):
    rng = np.random.default_rng()
    A = rng.standard_normal((n, n)) + 1.0j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q

options = {
"dt": 0.005,
"n_eql": 1,
"n_ene_blocks": 1,
"n_sr_blocks": 5,
"n_blocks": 4,
"n_prop_steps": 50,
"n_walkers": 4,
"seed": 8,
"trial": "",
"walker_type": "",
}

def check_hf(mol, mf):
    nmo=np.shape(mf.mo_coeff)[-1]
    nelec=sum(mol.nelec)
    hcore_ao = hf.get_hcore(mol)
    n_ao = hcore_ao.shape[-1]

    # RHF/UHF
    pyscf_interface.prep_afqmc(mf, tmpdir=tmpdir, chol_cut=chol_cut)

    if isinstance(mf, scf.rhf.RHF):
        options["trial"] = "rhf"
        options["walker_type"] = "restricted"
    elif isinstance(mf, scf.uhf.UHF):
        options["trial"] = "uhf"
        options["walker_type"] = "unrestricted"
    
    ene1, err1 = run_afqmc.run_afqmc(options=options, mpi_prefix=None, nproc=None, tmpdir=tmpdir)
    # RHF/UHF based GHF
    mo_coeff = mf.mo_coeff
    mo_coeff = transform_matrix(mo_coeff)

    pyscf_interface.prep_afqmc_ghf_complex(mol, mo_coeff+0.0j, hcore_ao, n_ao, tmpdir, chol_cut=chol_cut)

    options["trial"] = "ghf_complex"
    options["walker_type"] = "generalized"   
 
    ene2, err2 = run_afqmc.run_afqmc(options=options, mpi_prefix=None, nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} {ene2}"
    assert np.isclose(err1, err2, atol=1e-8), f"{err1} {err2}"
    
    # RHF/UHF based GHF with a complex rotation
    mo_coeff = mf.mo_coeff
    mo_coeff = transform_matrix(mo_coeff)
    mo_coeff = mo_coeff @ la.block_diag(random_orthogonal_complex(nelec), random_orthogonal_complex(2*nmo-nelec))

    pyscf_interface.prep_afqmc_ghf_complex(mol, mo_coeff+0j, hcore_ao, n_ao, tmpdir, chol_cut=chol_cut)
    
    ene2, err2 = run_afqmc.run_afqmc(options=options, mpi_prefix=None, nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} {ene2}"
    assert np.isclose(err1, err2, atol=1e-8), f"{err1} {err2}"

def check_cc(mol, mf):
    nmo=np.shape(mf.mo_coeff)[-1]
    nelec=sum(mol.nelec)
    hcore_ao = hf.get_hcore(mol)
    n_ao = hcore_ao.shape[-1]

    # CCSD
    mycc = cc.CCSD(mf)
    mycc.kernel()

    pyscf_interface.prep_afqmc(mycc, tmpdir=tmpdir, chol_cut=chol_cut)

    if isinstance(mf, scf.rhf.RHF):
        options["trial"] = "cisd"
        options["walker_type"] = "restricted"
    elif isinstance(mf, scf.uhf.UHF):
        options["trial"] = "ucisd"
        options["walker_type"] = "unrestricted"

    ene1, err1 = run_afqmc.run_afqmc(options=options, mpi_prefix=None, nproc=None, tmpdir=tmpdir)

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
    
    ene2, err2 = run_afqmc.run_afqmc(options=options, mpi_prefix=None, nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} vs {ene2}"
    assert np.isclose(err1, err2, atol=1e-8), f"{err1} {err2}"

    # GCCSD with a complex rotation
    mo_coeff = gmf.mo_coeff
    U_occ = random_orthogonal_complex(nelec)
    U_vir = random_orthogonal_complex(2*nmo-nelec)
    U = la.block_diag(U_occ, U_vir)
    gmf.mo_coeff = mo_coeff @ U

    ccsd.t1 = np.einsum("ia,ip,aq", ccsd.t1, U_occ, U_vir.conj())
    ccsd.t2 = np.einsum("ijab,ip,jq,ar,bs->pqrs", ccsd.t2, U_occ, U_occ, U_vir.conj(), U_vir.conj())
    np.savez(tmpdir+"/amplitudes.npz",t1=ccsd.t1, t2=ccsd.t2)

    pyscf_interface.prep_afqmc_ghf_complex(mol, gmf.mo_coeff+0j, hcore_ao, n_ao, tmpdir, chol_cut=chol_cut)

    ene2, err2 = run_afqmc.run_afqmc(options=options, mpi_prefix=None, nproc=None, tmpdir=tmpdir)
    assert np.isclose(ene1, ene2, atol=1e-6), f"{ene1} vs {ene2}"
    assert np.isclose(err1, err2, atol=1e-8), f"{err1} {err2}"
    
# H4
def test_h4():
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
def test_h2o():
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

# NH2 
def test_nh2():
    mol = gto.M(atom='''
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        ''',
        basis='sto-3g',
        spin=1,
        verbose = 3)
    mf = scf.UHF(mol)
    mf.kernel()

    check_hf(mol, mf)
    check_cc(mol, mf)

if __name__ =="__main__":
    test_h4()
    test_h2o()
    test_nh2()
