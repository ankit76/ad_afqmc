import numpy as np
from pyscf import __config__, ao2mo, df, dft, lib, mcscf, scf, lo
from ad_afqmc.pyscf_interface import *
import re


def run_afqmc_lno_mf(
    mf,
    integrals=None,
    norb_act=None,
    nelec_act=None,
    mo_coeff=None,
    norb_frozen=[],
    nproc=None,
    chol_cut=1e-5,
    seed=None,
    dt=0.005,
    nwalk_per_proc=5,
    nblocks=1000,
    orbitalE=-2,
    maxError=1e-4,
    prjlo=None,
    tmpdir="./",
    output_file_name="afqmc_output.out",
    n_eql=2,
    n_ene_blocks=25,
    n_sr_blocks=2,
):
    # print("#\n# Preparing AFQMC calculation")
    options = {
        "n_eql": n_eql,
        "n_ene_blocks": n_ene_blocks,
        "n_sr_blocks": n_sr_blocks,
        "n_blocks": nblocks,
        "n_walkers": nwalk_per_proc,
        "seed": seed,
        "walker_type": "rhf",
        "trial": "rhf",
        "dt": dt,
        "ad_mode": None,
        "orbE": orbitalE,
        "prjlo": prjlo,
        "maxError": maxError,
        "tmpdir": tmpdir,
    }
    import pickle

    with open("options.bin", "wb") as f:
        pickle.dump(options, f)

    mol = mf.mol
    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    # calculate cholesky integrals
    # print("# Calculating Cholesky integrals")
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    if integrals is not None:
        enuc = integrals["h0"]
        h1e = integrals["h1"]
        eri = integrals["h2"]
        nelec = mol.nelec
        nbasis = h1e.shape[-1]
        norb = nbasis
        eri = ao2mo.restore(4, eri, norb)
        chol0 = modified_cholesky(eri, chol_cut)
        nchol = chol0.shape[0]
        chol = np.zeros((nchol, norb, norb))
        for i in range(nchol):
            for m in range(norb):
                for n in range(m + 1):
                    triind = m * (m + 1) // 2 + n
                    chol[i, m, n] = chol0[i, triind]
                    chol[i, n, m] = chol0[i, triind]

    else:
        DFbas = mf.with_df.auxmol.basis
        h1e, chol, nelec, enuc = generate_integrals(
            mol, mf.get_hcore(), mo_coeff, chol_cut, DFbas=DFbas
        )
        nbasis = h1e.shape[-1]
        nelec = mol.nelec

        mc = mcscf.CASSCF(mf, norb_act, nelec_act)
        mc.frozen = norb_frozen
        nelec = mc.nelecas
        mc.mo_coeff = mo_coeff
        h1e, enuc = mc.get_h1eff()
        #     import pdb;pdb.set_trace()
        nbasis = mo_coeff.shape[-1]
        act = [i for i in range(nbasis) if i not in norb_frozen]
        e = ao2mo.kernel(mf.mol, mo_coeff[:, act], compact=False)
        chol = modified_cholesky(e, max_error=chol_cut)

    # print("# Finished calculating Cholesky integrals\n")

    nbasis = h1e.shape[-1]
    # print("# Size of the correlation space:")
    # print(f"# Number of electrons: {nelec}")
    # print(f"# Number of basis functions: {nbasis}")
    print(f"# Number of Cholesky vectors: {chol.shape[0]}\n")
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        hf_type = "uhf"
        uhfCoeffs = np.empty((nbasis, 2 * nbasis))
        if isinstance(mf, scf.uhf.UHF):
            q, r = np.linalg.qr(
                mo_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[0][:, norb_frozen:])
            )
            uhfCoeffs[:, :nbasis] = q
            q, r = np.linalg.qr(
                mo_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[1][:, norb_frozen:])
            )
            uhfCoeffs[:, nbasis:] = q
        else:
            q, r = np.linalg.qr(
                mo_coeff[:, norb_frozen:]
                .T.dot(overlap)
                .dot(mf.mo_coeff[:, norb_frozen:])
            )
            uhfCoeffs[:, :nbasis] = q
            uhfCoeffs[:, nbasis:] = q

        trial_coeffs[0] = uhfCoeffs[:, :nbasis]
        trial_coeffs[1] = uhfCoeffs[:, nbasis:]
        # np.savetxt("uhf.txt", uhfCoeffs)
        np.savez("mo_coeff.npz", mo_coeff=trial_coeffs)

    elif isinstance(mf, scf.rhf.RHF):
        hf_type = "rhf"
        # q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
        q = np.eye(mol.nao - len(norb_frozen))
        trial_coeffs[0] = q
        trial_coeffs[1] = q
        np.savez("mo_coeff.npz", mo_coeff=trial_coeffs)

    write_dqmc(
        h1e,
        h1e_mod,
        chol,
        sum(nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename="FCIDUMP_chol",
        mo_coeffs=trial_coeffs,
    )
    import os

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/launch_script.py"

    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREAD=1; mpirun -np {nproc} python {script} --use_mpi ./ >> {output_file_name}"
    )

    target_line_prefix = "Orbital energy: "
    with open(output_file_name, "r") as file:
        for line in file:
            if line.startswith(target_line_prefix):
                line = line[len(target_line_prefix) :].strip()
                values = line.split()
                value1 = float(values[0])
                if len(values) > 2:
                    value2 = values[2]
                    value2 = float(value2)
                else:
                    value2 = maxError

    input_file = output_file_name
    with open(input_file, "r") as infile:
        found_header = False
        for line in infile:
            if found_header:
                print(line.strip())
            elif line.strip().startswith("# Number of large deviations:"):
                found_header = True

    return value1, value2


def prep_local_orbitals(mf, frozen=0, localization_method="pm"):
    if localization_method not in ["pm"]:
        raise ValueError(
            f"Localization method '{localization_method}' is not supported. Make LOs by yourself."
        )
    orbocc = mf.mo_coeff[:, frozen : np.count_nonzero(mf.mo_occ)]
    mlo = lo.PipekMezey(mf.mol, orbocc)
    lo_coeff = mlo.kernel()
    while (
        True
    ):  # always performing jacobi sweep to avoid trapping in local minimum/saddle point
        lo_coeff1 = mlo.stability_jacobi()[1]
        if lo_coeff1 is lo_coeff:
            break
        mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    # Fragment list: for PM, every orbital corresponds to a fragment
    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

    return lo_coeff, frag_lolist
