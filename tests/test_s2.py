import os
import jax
import jax.numpy as jnp
from pyscf import gto, scf, fci, ci, cc
from pyscf.cc.uccsd import UCCSD
from ad_afqmc import afqmc, config, pyscf_interface, launch_script, Wigner_small_d
from ad_afqmc.walkers import UHFWalkers
import pickle

jax.config.update("jax_enable_x64", True)

def get_walker(key, nmo, nelec):
    n = nmo
    key1, key2 = jax.random.split(key)
    A = (
        jax.random.normal(key1, (n, n)) +
        1.0j * jax.random.normal(key2, (n, n))
    )
    Q, R = jnp.linalg.qr(A)
    Q *= jnp.sign(jnp.diag(R))

    return Q[:,:nelec]

mol =  gto.M(atom ="""
    N 0.0 0.0 0.0
    N 2.5 0.0 0.0
    """,
    basis = '6-31g',
    spin=0,
    verbose = 3)

# RHF
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)

path = os.path.dirname(os.path.abspath(__file__))
with open(path+"/test_s2.bin", "rb") as f:
    mf.mo_coeff = pickle.load(f)

mycc = cc.UCCSD(mf)
mycc.kernel()

def prep(obj, target_spin):
    af = afqmc.AFQMC(obj)
    af.chol_cut = 1e-5
    af.free_projection = True
    af.n_walkers = 1
    af.walker_type = "uhf"
    af.ene0 = obj.e_tot
    af.s2_projector_ngrid = 100
    af.symmetry_projector = "s2"
    af.target_spin = target_spin
    af.tmpdir = "."
    af.kernel(dry_run=True)
    
    with open("options.bin", "rb") as f:
        options = pickle.load(f)
    
    if isinstance(obj, UCCSD):
        pyscf_interface.read_pyscf_ccsd(obj, options["tmpdir"])

    (
        ham_data,
        ham,
        prop,
        trial_bra,
        wave_data_bra,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
    ) = launch_script.setup_afqmc_fp(options, options["tmpdir"])
    
    ham_data = ham.build_measurement_intermediates(ham_data, trial_bra, wave_data_bra)
    ham_data = ham.build_propagation_intermediates(
        ham_data, prop, trial_bra, wave_data_bra
    )

    return options, ham_data, trial_bra, wave_data_bra

def check(options, ham_data, trial_bra, wave_data_bra):

    # Random walker
    nmo = trial_bra.norb
    na, nb = trial_bra.nelec

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    wa = get_walker(subkey, nmo, na)
    key, subkey = jax.random.split(key)
    wb = get_walker(subkey, nmo, nb)

    walker = UHFWalkers([wa.reshape(1,nmo,na), wb.reshape(1, nmo, nb)])
    
    e1 = trial_bra.calc_energy(walker, ham_data, wave_data_bra)
    
    # Brute-force quadrature
    S = options["target_spin"] / 2.0
    Sz = (na - nb) / 2.0

    ngrid=1000
    betas = jnp.linspace(0, jnp.pi, ngrid, endpoint=False)
    w_betas = (
        jax.vmap(Wigner_small_d.wigner_small_d, (None, None, None, 0))(
            S, Sz, Sz, betas
        )
        * jnp.sin(betas)
        * (2 * S + 1)
        / 2.0
        * jnp.pi
        / ngrid
    )
    wave_data_bra["betas"] = (S, Sz, w_betas, betas)
    
    e2 = trial_bra.calc_energy(walker, ham_data, wave_data_bra)

    return e1, e2

def test_1():
    options, ham_data, trial_bra, wave_data_bra = prep(mf, 0.0)
    e1, e2 = check(options, ham_data, trial_bra, wave_data_bra)

    ref = -110.910853369073-1.040752960653j

    assert abs(e1-e2) < 1e-5
    assert abs(e2-ref) < 1e-5

def test_2():
    options, ham_data, trial_bra, wave_data_bra = prep(mf, 2.0)
    e1, e2 = check(options, ham_data, trial_bra, wave_data_bra)

    ref = -108.396448807131+1.194299917452j

    assert abs(e1-e2) < 1e-5
    assert abs(e2-ref) < 1e-5

def test_3():
    options, ham_data, trial_bra, wave_data_bra = prep(mf, 4.0)
    e1, e2 = check(options, ham_data, trial_bra, wave_data_bra)

    ref = -109.163625076631+0.524019304119j

    assert abs(e1-e2) < 1e-5
    assert abs(e2-ref) < 1e-5

def test_4():
    options, ham_data, trial_bra, wave_data_bra = prep(mycc, 0.0)
    e1, e2 = check(options, ham_data, trial_bra, wave_data_bra)

    ref = -109.068415590731+0.710889553071j

    assert abs(e1-e2) < 1e-5
    assert abs(e2-ref) < 1e-5

def test_5():
    options, ham_data, trial_bra, wave_data_bra = prep(mycc, 2.0)
    e1, e2 = check(options, ham_data, trial_bra, wave_data_bra)

    ref = -108.554877654471-0.040194320498j
    
    assert abs(e1-e2) < 1e-5
    assert abs(e2-ref) < 1e-5

def test_6():
    options, ham_data, trial_bra, wave_data_bra = prep(mycc, 4.0)
    e1, e2 = check(options, ham_data, trial_bra, wave_data_bra)

    ref = -108.304211594672-0.042412369043j

    assert abs(e1-e2) < 1e-5
    assert abs(e2-ref) < 1e-5

if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
