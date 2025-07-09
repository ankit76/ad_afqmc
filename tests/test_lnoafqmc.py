import numpy as np
import pytest
from ad_afqmc import config, lnoutils
from pyscf import gto, scf, df

config.setup_jax()

__test__ = False

options = {
    "dt": 0.005,
    "n_eql": 2,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 50,
    "n_prop_steps": 50,
    "n_walkers": 5,
    "seed": 8,
    "ad_mode": None,
    "trial": "rhf",
    "walker_type": "restricted",
}


# The actual check
def run_check(
    obj,
    mo_coeff,
    norb_act,
    nelec_act,
    norb_frozen,
    chol_cut,
    options,
    expected_energy,
    expected_error,
    atol,
    mpi,
):
    if mpi:
        nproc = 2
    else:
        nproc = 1
    e_corr_orb, err_corr_orb = lnoutils.run_afqmc_lno_mf(
        obj,
        norb_act=norb_act,
        nelec_act=nelec_act,
        mo_coeff=mo_coeff,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        seed=options["seed"],
        dt=options["dt"],
        nwalk_per_proc=options["n_walkers"],
        nblocks=options["n_blocks"],
        orbitalE=expected_energy,
        maxError=options["maxError"],
        prjlo=options["prjlo"],
        tmpdir=options["tmpdir"],
        output_file_name="afqmc_output.out",
        n_eql=options["n_eql"],
        n_ene_blocks=options["n_ene_blocks"],
        n_sr_blocks=options["n_sr_blocks"],
        nproc=nproc,
    )
    print(f"Energy correction: {e_corr_orb} +/- {err_corr_orb}")

    assert np.isclose(e_corr_orb, expected_energy, atol)
    assert np.allclose(err_corr_orb, expected_error, atol=atol)


def test_h2o_dimer():

    atol = 1e-5

    r = 1.5
    atomstring = """
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
        """
    mol = gto.M(atom=atomstring, basis="ccpvdz", unit="Bohr")
    mf_rhf = df.density_fit(scf.RHF(mol))
    mf_rhf.kernel()

    prjlo = np.array([[0.8, 0.0, 0.5, 0.0]])
    prjlo = prjlo / np.linalg.norm(prjlo)

    options["prjlo"] = prjlo
    norb_act = 18
    nelec_act = 8
    norb_frozen = [i for i in range(6)] + [i for i in range(24, 48)]
    chol_cut = 1e-4
    options["tmpdir"] = "./"
    options["maxError"] = 1e-5

    orbital_corr = -0.009938226221
    orbital_corr_error = 0.000482954279

    run_check(
        mf_rhf,
        mo_coeff=mf_rhf.mo_coeff,
        norb_act=norb_act,
        nelec_act=nelec_act,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        options=options,
        expected_energy=orbital_corr,
        expected_error=orbital_corr_error,
        atol=atol,
        mpi=False,
    )

    orbital_corr_mpi = -0.009829596951
    orbital_corr_error_mpi = 0.0003276177411

    run_check(
        mf_rhf,
        mo_coeff=mf_rhf.mo_coeff,
        norb_act=norb_act,
        nelec_act=nelec_act,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        options=options,
        expected_energy=orbital_corr_mpi,
        expected_error=orbital_corr_error_mpi,
        atol=atol,
        mpi=True,
    )


if __name__ == "__main__":
    test_h2o_dimer()
