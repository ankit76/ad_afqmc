import numpy as np
import pytest
from ad_afqmc import config, pyscf_interface, run_afqmc, grad_utils
from pyscf import gto, scf, df

config.setup_jax()


options = {
    "dt": 0.005,
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 20,
    "n_prop_steps": 50,
    "n_walkers": 5,
    "seed": 8,
    "ad_mode": "nuc_grad",
}


# The actual check
def run_check(obj, options, expected_energy, expected_grad, atol, mpi):
    tmpdir = "./"
    grad_utils.prep_afqmc_nuc_grad(obj, dR=1e-5, tmpdir=tmpdir)

    # mpi_prefix = "mpirun" if mpi else None
    nproc = 2 if mpi else None

    ene, _ = run_afqmc.run_afqmc(options=options, nproc=nproc, tmpdir=tmpdir)
    grad, grad_err = grad_utils.calculate_nuc_gradients(tmpdir, printG=False)

    assert np.isclose(ene, expected_energy, atol)
    assert np.allclose(grad, expected_grad, atol=atol)


def test_h2():

    atol = 1e-5

    r = 1.5
    atomstring = f"H 0 0 0; H 0 0 {r}"
    mol = gto.M(atom=atomstring, basis="631g", unit="Bohr")
    mf_rhf = df.density_fit(scf.RHF(mol))
    mf_rhf.kernel()

    # RHF
    rhf_1coreE = -1.15375239
    rhf_1coreG = np.array(
        [[-0.00518736, -0.00095493, -0.028576], [0.00518736, 0.00095493, 0.028576]]
    )
    options["trial"] = "rhf"
    options["walker_type"] = "restricted"
    run_check(mf_rhf, options, rhf_1coreE, rhf_1coreG, atol, mpi=False)

    # RHF with mpi
    rhf_2coreE = -1.15314153
    rhf_2coreG = [
        [-0.00347243, -0.00082577, -0.02950069],
        [0.00347243, 0.00082577, 0.02950069],
    ]
    run_check(mf_rhf, options, rhf_2coreE, rhf_2coreG, atol, mpi=True)

    # UHF
    mf_uhf = df.density_fit(scf.UHF(mol))
    mf_uhf.kernel()
    uhf_1coreE = -1.15375239
    uhf_1coreG = np.array(
        [
            [-0.00518736, -0.00095493, -0.028576],
            [0.00518736, 0.00095493, 0.028576],
        ]
    )
    options["trial"] = "uhf"
    options["walker_type"] = "unrestricted"
    run_check(mf_uhf, options, uhf_1coreE, uhf_1coreG, atol, mpi=False)

    # UHF with mpi
    uhf_2coreE = -1.15314153
    uhf_2coreG = np.array(
        [[-0.00347243, -0.00082577, -0.02950069], [0.00347243, 0.00082577, 0.02950069]]
    )
    run_check(mf_uhf, options, uhf_2coreE, uhf_2coreG, atol, mpi=True)

    # RHF no orbital rotation
    options["orbital_rotation"] = False
    rhf_no_sr_1coreE = -1.15375239
    rhf_no_sr_1coreG = np.array(
        [[-0.00518736, -0.00095493, -0.0279356], [0.00518736, 0.00095493, 0.0279356]]
    )
    options["trial"] = "rhf"
    options["walker_type"] = "restricted"
    run_check(mf_rhf, options, rhf_no_sr_1coreE, rhf_no_sr_1coreG, atol, mpi=False)


if __name__ == "__main__":
    test_h2()
