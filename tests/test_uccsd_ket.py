import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf
from ad_afqmc import afqmc, launch_script, pyscf_interface, wavefunctions
import pickle

def check_1(mf, mycc):
    af = afqmc.AFQMC(mf, mycc)
    af.free_projection = True
    af.dt= 0.1
    af.n_prop_steps= 10  # number of dt long steps in a propagation block
    af.n_blocks= 8  # number of propagation and measurement blocks
    af.n_ene_blocks= 10  # number of trajectories
    af.n_walkers= 5000
    af.walker_type= "unrestricted"
    af.ene0= mf.e_tot
    af.seed = 5
    af.kernel(dry_run=True)
    
    with open("tmpdir.txt", "r") as f:
        tmpdir = f.read()
    with open(tmpdir + "/options.bin", "rb") as f:
        options = pickle.load(f)
    
    pyscf_interface.read_pyscf_ccsd(mycc, tmpdir)
    
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
    
    init_walkers = None
    prop_data = prop.init_prop_data(
        trial_bra, wave_data_bra, ham_data, af.seed, init_walkers
    )
    
    e = []
    for i in range(5):
        wave_data_ket["key"] = jax.random.PRNGKey(af.seed+i)
        prop_data["walkers"] = trial_ket.get_init_walkers(
            wave_data_ket,
            af.n_walkers,
            "unrestricted",
        )
        
        energy_samples = jnp.real(
            trial_bra.calc_energy(prop_data["walkers"], ham_data, wave_data_bra)
        )
        e.append(jnp.array(jnp.sum(energy_samples) / af.n_walkers))
    
    e = np.mean(e)
    print(e)
    assert abs(e - mycc.e_tot) < 1e-3

def check_2(mf, mycc):
    af = afqmc.AFQMC(mf, mycc)
    af.free_projection = True
    af.dt= 0.1
    af.n_prop_steps= 2  # number of dt long steps in a propagation block
    af.n_blocks= 2  # number of propagation and measurement blocks
    af.n_ene_blocks= 2  # number of trajectories
    af.n_walkers= 2
    af.walker_type= "unrestricted"
    af.ene0= mf.e_tot
    af.seed = 5
    af.kernel()

def check_3(mycc):
    af = afqmc.AFQMC(mycc, mycc)
    af.free_projection = True
    af.dt= 0.1
    af.n_prop_steps= 2  # number of dt long steps in a propagation block
    af.n_blocks= 2  # number of propagation and measurement blocks
    af.n_ene_blocks= 2  # number of trajectories
    af.n_walkers= 2
    af.walker_type= "unrestricted"
    af.ene0= mycc.e_tot
    af.seed = 5
    af.kernel()

def run(mol):
    mf = scf.UHF(mol)
    mf.kernel()
    mycc = mf.CCSD()
    mycc.kernel()
    return mf, mycc

def test_s0():
    mol =  gto.M(atom ="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis = '6-31g',
        verbose = 3)

    mf, mycc = run(mol)
    check_1(mf, mycc)
    check_2(mf, mycc)

def test_s2():
    mol =  gto.M(atom ="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis = '6-31g',
        spin = 2,
        verbose = 3)

    mf, mycc = run(mol)
    check_1(mf, mycc)
    check_2(mf, mycc)

if __name__ == "__main__":
    test_s0()
    test_s2()
