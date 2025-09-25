import argparse
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import jax
import numpy as np
from jax import numpy as jnp

from ad_afqmc import (
    Wigner_small_d,
    config,
    driver,
    hamiltonian,
    propagation,
    sampling,
    wavefunctions,
)
from ad_afqmc.config import mpi_print as print

tmpdir = "."


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        args: Namespace containing the parsed arguments (tmpdir, use_gpu, use_mpi)
    """
    parser = argparse.ArgumentParser(description="Run AD-AFQMC calculation")
    parser.add_argument("tmpdir", help="Directory for input/output files")
    parser.add_argument(
        "--use_gpu", action="store_true", help="Enable GPU acceleration"
    )
    parser.add_argument("--use_mpi", action="store_true", help="Enable MPI parallelism")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    """
    Configure the computation environment based on command line arguments.

    Args:
        args: Parsed command line arguments
    """
    global tmpdir

    if args.use_gpu:
        config.afqmc_config["use_gpu"] = True
    if args.use_mpi:
        assert config.afqmc_config["use_gpu"] is False, "Inter GPU MPI not supported."
        config.afqmc_config["use_mpi"] = True
    tmpdir = args.tmpdir


def read_fcidump(tmp_dir: Optional[str] = None) -> Tuple:
    """
    Read the FCIDUMP_chol file containing Hamiltonian elements.

    Args:
        tmp_dir: Directory containing the FCIDUMP_chol file

    Returns:
        Tuple containing:
            h0: Core energy
            h1: One-body Hamiltonian matrix
            chol: Cholesky vectors
            norb: Number of molecular orbitals
            nelec_sp: Tuple of electrons per spin (alpha, beta)
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir

    assert os.path.isfile(
        directory + "/FCIDUMP_chol"
    ), f"File '{directory}/FCIDUMP_chol' does not exist."
    with h5py.File(directory + "/FCIDUMP_chol", "r") as fh5:
        [nelec, norb, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore")).reshape(norb, norb)
        chol = jnp.array(fh5.get("chol")).reshape(-1, norb, norb)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(norb) is np.int64
    ms, nelec, norb = int(ms), int(nelec), int(norb)

    # Calculate electrons per spin channel
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    return h0, h1, chol, norb, nelec_sp


def read_options(options: Optional[Dict] = None, tmp_dir: Optional[str] = None) -> Dict:
    """
    Read calculation options from file or use provided options with defaults.

    Args:
        options: Dictionary of options (if None, tries to load from file)
        tmp_dir: Directory containing the options.bin file

    Returns:
        Dictionary of calculation options with defaults applied
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir
    # Try to load options from file if not provided
    if options is None:
        try:
            with open(directory + "/options.bin", "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    # Set default values for options
    options["dt"] = options.get("dt", 0.01)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["n_ene_blocks_eql"] = options.get("n_ene_blocks_eql", 5)
    options["n_sr_blocks_eql"] = options.get("n_sr_blocks_eql", 10)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)

    # AD mode options
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm", "nuc_grad"]

    # Wavefunction and algorithm options
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "restricted")
    options["symmetry_projector"] = options.get("symmetry_projector", None)
    options["trial_ket"] = options.get("trial_ket", None)

    # Handle backwards compatibility for walker types
    if options["walker_type"] == "rhf":
        options["walker_type"] = "restricted"
    elif options["walker_type"] == "uhf":
        options["walker_type"] = "unrestricted"
    assert options["walker_type"] in ["restricted", "unrestricted", "generalized"]

    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    assert options["trial"] in [
        None,
        "rhf",
        "uhf",
        "noci",
        "cisd",
        "ucisd",
        "ghf_complex",
        "gcisd_complex",
        "UCISD",
    ]

    if options["trial"] is None:
        print(f"# No trial specified in options.")

    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)

    # performance and memory options
    options["n_chunks"] = options.get("n_chunks", options.get("n_batch", 1))
    options["vhs_mixed_precision"] = options.get("vhs_mixed_precision", False)
    options["trial_mixed_precision"] = options.get("trial_mixed_precision", False)
    options["memory_mode"] = options.get("memory_mode", "low")

    # LNO options
    options["prjlo"] = options.get("prjlo", None)
    options["orbE"] = options.get("orbE", 0)
    options["maxError"] = options.get("maxError", 1e-3)

    assert options is not None, "Options dictionary cannot be None."
    return options


def read_observable(
    nmo: int, options: Dict, tmp_dir: Optional[str] = None
) -> Optional[List]:
    """
    Read observable operator from file.

    Args:
        nmo: Number of molecular orbitals
        options: Dictionary of calculation options
        tmp_dir: Directory containing the observable.h5 file

    Returns:
        List containing observable operator and constant, or None if file not found
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir

    try:
        with h5py.File(directory + "/observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            if options["walker_type"] == "unrestricted":
                observable_op = jnp.array([observable_op, observable_op])
            observable = [observable_op, observable_constant]
    except:
        observable = None

    return observable


def set_ham(
    norb: int,
    h0: jnp.ndarray,
    h1: jnp.ndarray,
    chol: jnp.ndarray,
    ene0: float = 0.0,
) -> Tuple[Any, Dict]:
    """
    Set up the Hamiltonian for AFQMC calculation.

    Args:
        norb: Number of molecular orbitals
        h0: Core energy
        h1: One-body Hamiltonian matrix
        chol: Cholesky vectors
        ene0: Energy offset

    Returns:
        Tuple containing:
            ham: Hamiltonian object
            ham_data: Dictionary of Hamiltonian data
    """
    ham = hamiltonian.hamiltonian(norb)
    nchol = chol.shape[0]
    ham_data = {
        "h0": h0,
        "h1": jnp.array([h1, h1]),  # Replicate for up/down spins
        "chol": chol.reshape(nchol, -1),
        "ene0": ene0,
    }
    return ham, ham_data


def apply_symmetry_mask(ham_data: Dict, options: Dict) -> Dict:
    """
    Apply symmetry mask to Hamiltonian data based on options.

    Args:
        ham_data: Dictionary of Hamiltonian data
        options: Dictionary of calculation options

    Returns:
        Updated ham_data with mask applied
    """
    if options["symmetry"]:
        ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
    else:
        ham_data["mask"] = jnp.ones(ham_data["h1"].shape)

    return ham_data


def set_trial(
    options: Dict,
    options_trial: str,
    mo_coeff: jnp.ndarray,
    norb: int,
    nelec_sp: Tuple[int, int],
    tmp_dir: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """
    Set up the trial wavefunction.

    Args:
        options: Dictionary of calculation options
        mo_coeff: Molecular orbital coefficients
        norb: Number of orbitals
        nelec_sp: Tuple of electrons per spin (alpha, beta)
        tmp_dir: Directory containing wavefunction files

    Returns:
        Tuple containing:
            trial: Trial wavefunction object
            wave_data: Dictionary of wavefunction data
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir
    wave_data = {}

    # Try to read RDM1 from file
    try:
        rdm1 = jnp.array(np.load(directory + "/rdm1.npz")["rdm1"])
        assert rdm1.shape == (2, norb, norb)
        wave_data["rdm1"] = rdm1
        print(f"# Read RDM1 from disk")
    except:
        # Construct RDM1 from mo_coeff if file not found
        if options_trial in ["ghf_complex", "gcisd_complex"]:
            wave_data["rdm1"] = jnp.array(
                [
                    mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]
                    @ mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]].T.conj(),
                    mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]
                    @ mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]].T.conj(),
                ]
            )

        else:
            wave_data["rdm1"] = jnp.array(
                [
                    mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
                    mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
                ]
            )

    if options.get("symmetry_projector", None) == "s2":
        from numpy.polynomial.legendre import leggauss

        # Gauss-Legendre
        #
        # \int_0^\pi sin(\beta) f(\beta) \dd\beta
        # x = \cos(beta), \dd x = -\sin(beta)
        # = \int_{-1}^{1} f(\arccos(x)) \dd x
        #\approx \sum_{i=1}^n w_i f(\arccos(x_i))
        #
        S = options["target_spin"] / 2.0
        Sz = (nelec_sp[0] - nelec_sp[1]) / 2.0
        ngrid = options.get("ngrid")  # number of points for the quadrature

        # Gauss–Legendre
        x, w = leggauss(ngrid)
        beta_vals = jnp.arccos(x)  # map [-1, 1] to [0, pi]

        # Wigner small-d matrix elements for each point
        wigner_d = jax.vmap(
            Wigner_small_d.wigner_small_d, (None, None, None, 0)
        )(S, Sz, Sz, beta_vals)

        # Combine
        prefactor = 0.5 * (2*S + 1)
        wigner = prefactor * w * wigner_d
        wave_data["wigner"] = (S, Sz, wigner, beta_vals)

    # Set up trial wavefunction based on specified type
    if options_trial == "rhf":
        trial = wavefunctions.rhf(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    elif options_trial == "uhf":
        trial = wavefunctions.uhf(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]

    elif options_trial == "noci":
        with open(directory + "/dets.pkl", "rb") as f:
            ci_coeffs_dets = pickle.load(f)
        ci_coeffs_dets = [
            jnp.array(ci_coeffs_dets[0]),
            [jnp.array(ci_coeffs_dets[1][0]), jnp.array(ci_coeffs_dets[1][1])],
        ]
        wave_data["ci_coeffs_dets"] = ci_coeffs_dets
        trial = wavefunctions.noci(
            norb,
            nelec_sp,
            ci_coeffs_dets[0].size,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )

    elif options_trial == "cisd":
        try:
            amplitudes = np.load(directory + "/amplitudes.npz")
            ci1 = jnp.array(amplitudes["ci1"])
            ci2 = jnp.array(amplitudes["ci2"])
            trial_wave_data = {"ci1": ci1, "ci2": ci2}
            wave_data.update(trial_wave_data)

            if options["trial_mixed_precision"]:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128

            trial = wavefunctions.cisd(
                norb,
                nelec_sp,
                n_chunks=options["n_chunks"],
                projector=options["symmetry_projector"],
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options["memory_mode"],
            )
        except:
            raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")

    elif options_trial == "ccsd":
        assert options["walker_type"] == "restricted"
        try:
            amplitudes = np.load(directory + "/amplitudes.npz")
            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t1"])
            nocc, nvirt = t1.shape

            trial_wave_data = {
                "t1": t1,
            }

            wave_data.update(trial_wave_data)
            wave_data["mo_coeff"] = mo_coeff[0]

            if options["trial_mixed_precision"]:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128

            trial = wavefunctions.ccsd(
                norb,
                nelec_sp,
                nocc,
                nvirt,
                n_chunks=options["n_chunks"],
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options["memory_mode"],
            )
            wave_data = trial.hs_op(wave_data, amplitudes["t2"])
        except:
            raise ValueError("Trial specified as ccsd, but amplitudes.npz not found.")

    elif options_trial == "uccsd":
        assert options["walker_type"] == "unrestricted"
        try:
            amplitudes = np.load(directory + "/amplitudes.npz")

            trial_wave_data = {
                "t1a": amplitudes["t1a"],
                "t1b": amplitudes["t1b"],
            }

            nOa, nVa = amplitudes["t1a"].shape
            nOb, nVb = amplitudes["t1b"].shape
            nocc = (nOa, nOb)
            nvir = (nVa, nVb)

            assert nocc == nelec_sp
            assert nvir == (norb-nOa, norb-nOb)

            wave_data.update(trial_wave_data)
            wave_data["mo_coeff"] = [mo_coeff[0], mo_coeff[1]]

            if options["trial_mixed_precision"]:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128

            trial = wavefunctions.uccsd(
                norb,
                nelec_sp,
                nocc,
                nvir,
                n_chunks=options["n_chunks"],
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options["memory_mode"],
            )

            wave_data = trial.hs_op(wave_data,
                amplitudes["t2aa"],
                amplitudes["t2ab"],
                amplitudes["t2bb"],
            )
        except:
            raise ValueError("Trial specified as uccsd, but amplitudes.npz not found.")

    elif options_trial == "ucisd" or options_trial == "UCISD":
        try:
            amplitudes = np.load(directory + "/amplitudes.npz")
            ci1a = jnp.array(amplitudes["ci1a"])
            ci1b = jnp.array(amplitudes["ci1b"])
            ci2aa = jnp.array(amplitudes["ci2aa"])
            ci2ab = jnp.array(amplitudes["ci2ab"])
            ci2bb = jnp.array(amplitudes["ci2bb"])
            trial_wave_data = {
                "ci1A": ci1a,
                "ci1B": ci1b,
                "ci2AA": ci2aa,
                "ci2AB": ci2ab,
                "ci2BB": ci2bb,
                "mo_coeff": mo_coeff,
            }
            wave_data.update(trial_wave_data)

            if options["trial_mixed_precision"]:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128

            trial = (
                wavefunctions.ucisd(
                    norb,
                    nelec_sp,
                    n_chunks=options["n_chunks"],
                    projector=options["symmetry_projector"],
                    mixed_real_dtype=mixed_real_dtype,
                    mixed_complex_dtype=mixed_complex_dtype,
                    memory_mode=options["memory_mode"],
                )
                if options_trial == "ucisd"
                else wavefunctions.UCISD(
                    norb,
                    nelec_sp,
                    n_chunks=options["n_chunks"],
                    projector=options["symmetry_projector"],
                )
            )
        except:
            raise ValueError("Trial specified as ucisd, but amplitudes.npz not found.")

    elif options_trial == "ghf_complex":
        trial = wavefunctions.ghf_complex(norb, nelec_sp, n_chunks=options["n_chunks"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]

    elif options_trial == "gcisd_complex":
        try:
            amplitudes = np.load(directory + "/amplitudes.npz")

            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t2"])

            ci1 = t1
            ci2 = (
                np.einsum("ijab->iajb", t2)
                + np.einsum("ia,jb->iajb", t1, t1)
                - np.einsum("ib,ja->iajb", t1, t1)
            )
            trial_wave_data = {
                "ci1": ci1,
                "ci2": ci2,
                "mo_coeff": mo_coeff,
            }
            wave_data.update(trial_wave_data)
            trial = wavefunctions.gcisd_complex(
                norb, nelec_sp, n_chunks=options["n_chunks"]
            )
        except:
            raise ValueError(
                "Trial specified as gcisd_complex, but amplitudes.npz not found."
            )

    else:
        # Try to load trial from pickle file
        try:
            with open(directory + "/trial.pkl", "rb") as f:
                [trial, trial_wave_data] = pickle.load(f)
            wave_data.update(trial_wave_data)
            print(f"# Read trial of type {type(trial).__name__} from trial.pkl.")
        except:
            print("# trial.pkl not found, make sure to construct the trial separately.")
            trial = None

    if options["prjlo"] is not None:  # For LNO
        wave_data["prjlo"] = options["prjlo"]
        trial = wavefunctions.rhf_lno(norb, nelec_sp, n_chunks=options["n_chunks"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    return trial, wave_data


def set_prop(options: Dict) -> Any:
    """
    Set up the propagator for AFQMC calculation.

    Args:
        options: Dictionary of calculation options

    Returns:
        Propagator object configured according to options
    """
    vhs_real_dtype = jnp.float32 if options["vhs_mixed_precision"] else jnp.float64
    vhs_complex_dtype = (
        jnp.complex64 if options["vhs_mixed_precision"] else jnp.complex128
    )
    n_exp_terms = 10 if options["free_projection"] else options.get("n_exp_terms", 6)
    return propagation.propagator_afqmc(
        options["dt"],
        options["n_walkers"],
        n_exp_terms=n_exp_terms,
        n_chunks=options["n_chunks"],
        vhs_real_dtype=vhs_real_dtype,
        vhs_complex_dtype=vhs_complex_dtype,
        walker_type=options["walker_type"],
    )


def set_sampler(options: Dict) -> Any:
    """
    Set up the sampler for AFQMC calculation.

    Args:
        options: Dictionary of calculation options

    Returns:
        Sampler object configured according to options
    """
    if options["prjlo"] is not None:
        # Use LNO sampler if prjlo is specified
        return sampling.sampler_lno(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
        )
    else:
        return sampling.sampler(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
        )


def load_mo_coefficients(tmp_dir: Optional[str] = None) -> jnp.ndarray:
    """
    Load molecular orbital coefficients from file.

    Args:
        tmp_dir: Directory containing mo_coeff.npz file

    Returns:
        Array of molecular orbital coefficients
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir
    try:
        return jnp.array(np.load(directory + "/mo_coeff.npz")["mo_coeff"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find mo_coeff.npz in {directory}")
    except Exception as e:
        raise RuntimeError(f"Error loading molecular orbital coefficients: {str(e)}")


def setup_afqmc(
    options: Optional[Dict] = None,
    tmp_dir: Optional[str] = None,
) -> Tuple:
    """
    Prepare all components for an AFQMC calculation.

    Args:
        options: Dictionary of calculation options (optional)
        tmp_dir: Directory for input/output files (optional)

    Returns:
        Tuple containing all necessary components for AFQMC calculation:
            ham_data: Dictionary of Hamiltonian data
            ham: Hamiltonian object
            prop: Propagator object
            trial: Trial wavefunction object
            wave_data: Dictionary of wavefunction data
            sampler: Sampler object
            observable: Observable operator
            options: Dictionary of calculation options
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir

    h0, h1, chol, norb, nelec_sp = read_fcidump(directory)
    options = read_options(options, directory)
    observable = read_observable(norb, options, directory)
    ham, ham_data = set_ham(norb, h0, h1, chol, options["ene0"])
    ham_data = apply_symmetry_mask(ham_data, options)
    mo_coeff = load_mo_coefficients(directory)
    trial, wave_data = set_trial(
        options, options["trial"], mo_coeff, norb, nelec_sp, directory
    )
    trial_ket, wave_data_ket = set_trial(
        options,
        options.get("trial_ket", options["trial"]),
        mo_coeff,
        norb,
        nelec_sp,
        directory,
    )
    prop = set_prop(options)
    sampler = set_sampler(options)

    print(f"# norb: {norb}")
    print(f"# nelec: {nelec_sp}")
    print("#")
    for op in options:
        if options[op] is not None:
            print(f"# {op}: {options[op]}")
    print("#")

    return (
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
    )


def run_afqmc_calculation(
    mpi_comm: Optional[Any] = None,
    tmp_dir: Optional[str] = None,
    custom_options: Optional[Dict] = None,
) -> Tuple[float, float]:
    """
    Run the full AFQMC calculation.

    Args:
        mpi_comm: MPI communicator (optional)
        tmp_dir: Directory for input/output files (optional)
        custom_options: Custom calculation options (optional)

    Returns:
        Tuple containing:
            e_afqmc: AFQMC energy
            err_afqmc: Error in AFQMC energy
    """
    if mpi_comm is None:
        mpi_comm = config.setup_comm()

    directory = tmp_dir if tmp_dir is not None else tmpdir

    # Prepare all components
    ham_data, ham, prop, trial, wave_data, _, _, sampler, observable, options = (
        setup_afqmc(options=custom_options, tmp_dir=directory)
    )

    assert trial is not None, "Trial wavefunction is None. Cannot run AFQMC."

    # Initialize timing and synchronize MPI processes
    init_time = time.time()
    comm = mpi_comm.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()

    # Run appropriate AFQMC algorithm
    e_afqmc, err_afqmc = 0.0, 0.0
    if options["free_projection"]:
        # driver.fp_afqmc(
        #     ham_data,
        #     ham,
        #     prop,
        #     trial,
        #     wave_data,
        #     sampler,
        #     observable,
        #     options,
        #     mpi_comm,
        # )
        raise NotImplementedError(
            "Free projection AFQMC is not supported from launch_script."
        )
    else:
        e_afqmc, err_afqmc = driver.afqmc(
            ham_data,
            ham,
            prop,
            trial,
            wave_data,
            sampler,
            observable,
            options,
            mpi_comm,
            tmpdir=directory,
        )

    # Finalize timing and synchronize MPI processes
    comm.Barrier()
    end_time = time.time()

    # Print timing and save results
    print(f"AFQMC total walltime: {end_time - init_time}")
    if rank == 0:
        np.savetxt(directory + "/ene_err.txt", np.array([e_afqmc, err_afqmc]))

    comm.Barrier()
    return e_afqmc, err_afqmc


def main() -> None:
    """
    Main entry point for the lauch script.
    """
    args = parse_arguments()
    configure_environment(args)
    config.setup_jax()
    mpi_comm = config.setup_comm()
    run_afqmc_calculation(mpi_comm=mpi_comm, tmp_dir=args.tmpdir)


if __name__ == "__main__":
    main()
