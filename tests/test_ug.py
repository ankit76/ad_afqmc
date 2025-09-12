import os
import copy
import numpy as np
import scipy.linalg as la
import jax.numpy as jnp
from jax import config
from jax._src.typing import DTypeLike
import pickle
from pyscf import gto, scf, cc
from ad_afqmc import pyscf_interface, launch_script

config.update("jax_enable_x64", True)

np.set_printoptions(linewidth=1000, precision=6, suppress=True)
jnp.set_printoptions(linewidth=1000)

chol_cut = 1e-8
atol = 1e-8
tmp_ghf = "tmp_ghf"
tmp_uhf = "tmp_uhf"

os.makedirs(tmp_ghf, exist_ok=True)
os.makedirs(tmp_uhf, exist_ok=True)

def get_walker(nmo, nelec):
    n = nmo
    rng = np.random.default_rng()
    A = rng.standard_normal((n, n)) + 1.0j * rng.standard_normal((n, n))
    Q, R = jnp.linalg.qr(A)
    Q *= jnp.sign(jnp.diag(R))

    return Q[:,:nelec]

class Obj:
    pass

options = {
"dt": 0.005,
"n_eql": 0,
"n_ene_blocks": 1,
"n_sr_blocks": 1,
"n_blocks": 20,
"n_prop_steps": 50,
"n_walkers": 50,
"seed": 8,
"trial": "",
"walker_type": "",
"memory_mode": "high"
}

atomstring = f"""
N 0.0 0.0 0.0
N 2.5 0.0 0.0
#O     0.000000     0.000000     0.000000
#H     0.758602     0.000000     0.504284
#H    -0.758602     0.000000     0.504284
"""
mol = gto.M(atom=atomstring, basis="cc-pvdz", spin=0, unit="Angstrom", verbose=3)

# UHF
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)

# UHF -> GHF
gmf = scf.addons.convert_to_ghf(mf)

# UCCSD
ucc = cc.CCSD(mf)
ucc.kernel()

# UCCSD -> GCCSD
gcc = cc.addons.convert_to_gccsd(ucc)
np.savez(tmp_ghf+"/amplitudes.npz",t1=gcc.t1, t2=gcc.t2)

na, nb = mol.nelec
nelec = na+nb
nmo = np.shape(mf.mo_coeff)[-1]

# ghf
pyscf_interface.prep_afqmc_ghf_complex(mol, gmf, tmpdir=tmp_ghf, chol_cut=chol_cut)
ghf = Obj()
ghf.options = options
ghf.options["trial"] = "ghf_complex"
ghf.options["walker_type"] = "generalized"
ghf.ham_data, ghf.ham, ghf.prop, ghf.trial, ghf.wave_data, _, _, ghf.sampler, ghf.observable, ghf.options = launch_script.setup_afqmc(
    options=ghf.options, tmp_dir=tmp_ghf
)
ghf.ham_data = ghf.trial._build_measurement_intermediates(ghf.ham_data, ghf.wave_data)

# uhf
pyscf_interface.prep_afqmc(mf, chol_cut=chol_cut, tmpdir=tmp_uhf)
uhf = Obj()
uhf.options = options
uhf.options["trial"] = "uhf"
uhf.options["walker_type"] = "unrestricted"
uhf.ham_data, uhf.ham, uhf.prop, uhf.trial, uhf.wave_data, _, _, uhf.sampler, uhf.observable, uhf.options = launch_script.setup_afqmc(
    options=uhf.options, tmp_dir=tmp_uhf
)
uhf.ham_data = uhf.trial._build_measurement_intermediates(uhf.ham_data, uhf.wave_data)

# gcisd
pyscf_interface.prep_afqmc_ghf_complex(mol, gmf, tmpdir=tmp_ghf, chol_cut=chol_cut)
gcisd = Obj()
gcisd.options = options
gcisd.options["trial"] = "gcisd_complex"
gcisd.options["walker_type"] = "generalized"
gcisd.ham_data, gcisd.ham, gcisd.prop, gcisd.trial, gcisd.wave_data, _, _, gcisd.sampler, gcisd.observable, gcisd.options = launch_script.setup_afqmc(
    options=gcisd.options, tmp_dir=tmp_ghf
)

# ucisd
pyscf_interface.prep_afqmc(ucc, chol_cut=chol_cut, tmpdir=tmp_uhf)
ucisd = Obj()
ucisd.options = options
ucisd.options["trial"] = "ucisd"
ucisd.options["walker_type"] = "unrestricted"
ucisd.ham_data, ucisd.ham, ucisd.prop, ucisd.trial, ucisd.wave_data, _, _, ucisd.sampler, ucisd.observable, ucisd.options = launch_script.setup_afqmc(
    options=ucisd.options, tmp_dir=tmp_uhf
)
ucisd.ham_data = ucisd.trial._build_measurement_intermediates(ucisd.ham_data, ucisd.wave_data)
ucisd.trial._mixed_real_dtype_checking: DTypeLike = jnp.float64
ucisd.trial._mixed_complex_dtype_checking: DTypeLike = jnp.complex128

# w_ghf <-> w_uhf
# w_uhf = Y.T @ X @ w_ghf
# w_ghf = X.T @ Y @ w_uhf
# with X and Y s.t.
# uhf_mos X = ghf_mos 
# uhf_trial Y = ghf_trial

# Unitary transformation UHF MOs <-> GHF MOs
uhf_mos = la.block_diag(mf.mo_coeff[0], mf.mo_coeff[1])
ghf_mos = gmf.mo_coeff
# uhf_mos X = ghf_mos 
X = np.linalg.solve(uhf_mos, ghf_mos)

# Check
# X.T @ X = Id
res = np.linalg.norm(X.T @ X - np.identity(2*nmo))
assert res < 1e-14, res

# Unitary transformation UHF trial <-> GHF trial
uhf_trial = la.block_diag(ucisd.wave_data["mo_coeff"][0], ucisd.wave_data["mo_coeff"][1])
ghf_trial = gcisd.wave_data["mo_coeff"][0]
# uhf_trial Y = ghf_trial
Y = np.linalg.solve(uhf_trial, ghf_trial)

# Check
# Y.T @ Y = Id
res = np.linalg.norm(Y.T @ Y - np.identity(2*nmo))
assert res < 1e-14

# Define UHF and GHF trials for the phase
uhf_a = ucisd.wave_data["mo_coeff"][0][:,:na]
uhf_b = ucisd.wave_data["mo_coeff"][1][:,:nb]
uhf_trial = la.block_diag(uhf_a, uhf_b)
ghf_trial = gcisd.wave_data["mo_coeff"][0][:,:nelec]

def assert_ghf(w_ghf):
    res = np.linalg.norm(w_ghf - X.T @ Y @ Y.T @ X @ w_ghf)
    assert res < 1e-14

def assert_uhf(w_uhf):
    res = np.linalg.norm(w_uhf - Y.T @ X @ X.T @ Y @ w_uhf)
    assert res < 1e-14

def check_overlap_ghf_walker(w_ghf):
    w_uhf = Y.T @ X @ w_ghf
    assert_ghf(w_ghf)
    assert_uhf(w_uhf)
    phase = jnp.linalg.det(uhf_trial.T @ w_uhf) / jnp.linalg.det(ghf_trial.T @ w_ghf)
    ghf_o = ghf.trial._calc_overlap_generalized(w_ghf, ghf.wave_data)
    uhf_o = phase * uhf.trial._calc_overlap_generalized(w_uhf, uhf.wave_data)
    gcisd_o = gcisd.trial._calc_overlap_generalized(w_ghf, gcisd.wave_data)
    ucisd_o = phase * ucisd.trial._calc_overlap_generalized(w_uhf, ucisd.wave_data)

    print(f" <ghf|w_g>   = {ghf_o:.12f}")
    print(f" <uhf|w_g>   = {uhf_o:.12f}\n")
    print(f" <gcisd|w_g> = {gcisd_o:.12f}")
    print(f" <ucisd|w_u> = {ucisd_o:.12f}\n")

    assert np.isclose(ghf_o, uhf_o, atol=atol)
    assert np.isclose(gcisd_o, ucisd_o, atol=atol)

def check_energy_ghf_walker(w_ghf):
    w_uhf = Y.T @ X @ w_ghf
    assert_ghf(w_ghf)
    assert_uhf(w_uhf)
    ghf_e = ghf.trial._calc_energy_generalized(w_ghf, ghf.ham_data, ghf.wave_data)
    uhf_e = uhf.trial._calc_energy_generalized(w_uhf, uhf.ham_data, uhf.wave_data)
    gcisd_e = gcisd.trial._calc_energy_generalized(w_ghf, gcisd.ham_data, gcisd.wave_data)
    ucisd_e = ucisd.trial._calc_energy_generalized(w_uhf, ucisd.ham_data, ucisd.wave_data)

    print(f" E(ghf,w_g)   = {ghf_e:.12f}")
    print(f" E(uhf,w_u)   = {uhf_e:.12f}\n")
    print(f" E(gcisd,w_g) = {gcisd_e:.12f}")
    print(f" E(ucisd,w_u) = {ucisd_e:.12f}\n")

    assert np.isclose(ghf_e, uhf_e, atol=atol)
    assert np.isclose(gcisd_e, ucisd_e, atol=atol)

def check_overlap_uhf_walker(w_a, w_b):
    w_uhf = la.block_diag(w_a, w_b)
    w_ghf = X.T @ Y @ w_uhf
    assert_ghf(w_ghf)
    assert_uhf(w_uhf)
    phase = jnp.linalg.det(uhf_trial.T @ w_uhf) / jnp.linalg.det(ghf_trial.T @ w_ghf)
    ghf_o     = phase * ghf.trial._calc_overlap_generalized(w_ghf, ghf.wave_data)
    uhf_u_o   = uhf.trial._calc_overlap_unrestricted(w_a, w_b, uhf.wave_data) 
    uhf_g_o   = uhf.trial._calc_overlap_generalized(w_uhf, uhf.wave_data) 
    gcisd_o   = phase * gcisd.trial._calc_overlap_generalized(w_ghf, gcisd.wave_data)
    ucisd_u_o = ucisd.trial._calc_overlap_unrestricted(w_a, w_b, ucisd.wave_data) 
    ucisd_g_o = ucisd.trial._calc_overlap_generalized(w_uhf, ucisd.wave_data) 

    print(f" <ghf|w_g>    = {ghf_o:.12f}")
    print(f" <uhf|w_ab>   = {uhf_u_o:.12f}")
    print(f" <uhf|w_u>    = {uhf_g_o:.12f}\n")
    print(f" <gcisd|w_g>  = {gcisd_o:.12f}")
    print(f" <ucisd|w_ab> = {ucisd_u_o:.12f}")
    print(f" <ucisd|w_u>  = {ucisd_g_o:.12f}\n")

    assert np.isclose(ghf_o, uhf_u_o, atol=atol)
    assert np.isclose(ghf_o, uhf_g_o, atol=atol)
    assert np.isclose(gcisd_o, ucisd_u_o, atol=atol)
    assert np.isclose(gcisd_o, ucisd_g_o, atol=atol)

def check_energy_uhf_walker(w_a, w_b, display=True):
    w_uhf = la.block_diag(w_a, w_b)
    w_ghf = X.T @ Y @ w_uhf
    assert_ghf(w_ghf)
    assert_uhf(w_uhf)
    ghf_e   = ghf.trial._calc_energy_generalized(w_ghf, ghf.ham_data, ghf.wave_data)
    uhf_u_e   = uhf.trial._calc_energy_unrestricted(w_a, w_b, uhf.ham_data, uhf.wave_data)
    uhf_g_e   = uhf.trial._calc_energy_generalized(w_uhf, uhf.ham_data, uhf.wave_data)
    gcisd_e   = gcisd.trial._calc_energy_generalized(w_ghf, gcisd.ham_data, gcisd.wave_data)
    ucisd_u_e = ucisd.trial._calc_energy_unrestricted(w_a, w_b, ucisd.ham_data, ucisd.wave_data)
    ucisd_g_e = ucisd.trial._calc_energy_generalized(w_uhf, ucisd.ham_data, ucisd.wave_data)

    print(f" E(ghf,w_g)    = {ghf_e:.12f}")
    print(f" E(uhf,w_ab)   = {uhf_u_e:.12f}")
    print(f" E(uhf,w_u)    = {uhf_g_e:.12f}\n")
    print(f" E(gcisd,w_g)  = {gcisd_e:.12f}")
    print(f" E(ucisd,w_ab) = {ucisd_u_e:.12f}")
    print(f" E(ucisd,w_u)  = {ucisd_g_e:.12f}\n")

    assert np.isclose(ghf_e, uhf_u_e, atol=atol)
    assert np.isclose(ghf_e, uhf_g_e, atol=atol)
    assert np.isclose(gcisd_e, ucisd_u_e, atol=atol)
    assert np.isclose(gcisd_e, ucisd_g_e, atol=atol)

### Test 0
def test_0():
    print("\n### 0 ###")
    w_a, w_b = ucisd.wave_data["mo_coeff"][0][:,:na], ucisd.wave_data["mo_coeff"][1][:,:nb] 
    
    check_overlap_uhf_walker(w_a, w_b)
    check_energy_uhf_walker(w_a, w_b)
    #print(f" E(uccsd)      = {ucc.e_tot:.12f}")
    #print(f" E(gccsd)      = {gcc.e_tot:.12f}\n")
    
### Test 1
def test_1():
    print("\n### 1 ###")
    w_a = get_walker(nmo,nmo)[:,:na] 
    w_b = get_walker(nmo,nmo)[:,:nb]
    
    check_overlap_uhf_walker(w_a, w_b)
    check_energy_uhf_walker(w_a, w_b)
    
### Test 2
def test_2():
    print("\n### 2 ###")
    w_ghf = get_walker(2*nmo, nelec)
    
    check_overlap_ghf_walker(w_ghf)
    check_energy_ghf_walker(w_ghf)

if __name__ == "__main__":
    test_0()
    test_1()
    test_2()

