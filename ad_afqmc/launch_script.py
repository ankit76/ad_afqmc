import argparse
import os
import sys
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from jax import numpy as jnp

from ad_afqmc import config, driver, hamiltonian, propagation, sampling, wavefunctions
#from ad_afqmc.config import mpi_print as print
from ad_afqmc.options import Options
from ad_afqmc.logger import Logger

tmpdir = "."


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        args: Namespace containing the parsed arguments (tmpdir, verbose, use_gpu, use_mpi)
    """
    parser = argparse.ArgumentParser(description="Run AD-AFQMC calculation")
    parser.add_argument("tmpdir", help="Directory for input/output files")
    parser.add_argument("--verbose", default= 3, type=int, help="Verbose level")
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


def read_options(options: Optional[Options] = None, tmp_dir: Optional[str] = None) -> Options:
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
            raise FileNotFoundError

        options = Options.from_dict(options)

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


def apply_symmetry_mask(ham_data: Dict, options: Options) -> Dict:
    """
    Apply symmetry mask to Hamiltonian data based on options.

    Args:
        ham_data: Dictionary of Hamiltonian data
        options: Dictionary of calculation options

    Returns:
        Updated ham_data with mask applied
    """
    if options.symmetry:
        ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
    else:
        ham_data["mask"] = jnp.ones(ham_data["h1"].shape)

    return ham_data


def set_trial(
    log: Logger,
    options: Options,
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
        log.log_0(f"# Read RDM1 from disk")
    except:
        # Construct RDM1 from mo_coeff if file not found
        if options.trial in ["ghf_complex", "gcisd_complex"]:
            wave_data["rdm1"] = jnp.array(
            [
            mo_coeff[0][:, : nelec_sp[0]+nelec_sp[1]] @ mo_coeff[0][:, : nelec_sp[0]+nelec_sp[1]].T.conj(),
            mo_coeff[0][:, : nelec_sp[0]+nelec_sp[1]] @ mo_coeff[0][:, : nelec_sp[0]+nelec_sp[1]].T.conj(),
            ])

        else:
            wave_data["rdm1"] = jnp.array(
                [
                    mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
                    mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
                ]
            )

    # Set up trial wavefunction based on specified type
    if options.trial == "rhf":
        trial = wavefunctions.rhf(norb, nelec_sp, n_batch=options.n_batch)
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    elif options.trial == "uhf":
        trial = wavefunctions.uhf(norb, nelec_sp, n_batch=options.n_batch)
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]

    elif options.trial == "noci":
        with open(directory + "/dets.pkl", "rb") as f:
            ci_coeffs_dets = pickle.load(f)
        ci_coeffs_dets = [
            jnp.array(ci_coeffs_dets[0]),
            [jnp.array(ci_coeffs_dets[1][0]), jnp.array(ci_coeffs_dets[1][1])],
        ]
        wave_data["ci_coeffs_dets"] = ci_coeffs_dets
        trial = wavefunctions.noci(
            norb, nelec_sp, ci_coeffs_dets[0].size, n_batch=options.n_batch
        )

    elif options.trial == "cisd":
        try:
            amplitudes = np.load(directory + "/amplitudes.npz")
            ci1 = jnp.array(amplitudes["ci1"])
            ci2 = jnp.array(amplitudes["ci2"])
            trial_wave_data = {"ci1": ci1, "ci2": ci2}
            wave_data.update(trial_wave_data)

            if options.trial_mixed_precision:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128

            trial = wavefunctions.cisd(
                norb,
                nelec_sp,
                n_batch=options.n_batch,
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options.memory_mode,
            )
        except:
            raise ValueError("Trial specified as cisd, but amplitudes.npz not found.")

    elif options.trial == "ucisd":
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

            if options.trial_mixed_precision:
                mixed_real_dtype = jnp.float32
                mixed_complex_dtype = jnp.complex64
            else:
                mixed_real_dtype = jnp.float64
                mixed_complex_dtype = jnp.complex128

            trial = wavefunctions.ucisd(
                norb,
                nelec_sp,
                n_batch=options.n_batch,
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options.memory_mode,
            )
        except:
            raise ValueError("Trial specified as ucisd, but amplitudes.npz not found.")

    elif options.trial == "ghf_complex":
        trial = wavefunctions.ghf_complex(norb, nelec_sp, n_batch=options.n_batch)
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]+nelec_sp[1]]

    elif options.trial == "gcisd_complex":
        try:
            amplitudes = np.load(tmpdir + "/amplitudes.npz")

            t1 = jnp.array(amplitudes["t1"])
            t2 = jnp.array(amplitudes["t2"])

            ci1 = t1
            ci2 = np.einsum("ijab->iajb",t2) \
                + np.einsum("ia,jb->iajb",t1,t1) \
                - np.einsum("ib,ja->iajb",t1,t1)
            trial_wave_data = {
                "ci1": ci1,
                "ci2": ci2,
                "mo_coeff": mo_coeff,
            }
            wave_data.update(trial_wave_data)
            trial = wavefunctions.gcisd_complex(norb, nelec_sp, n_batch=options.n_batch)
        except:
            raise ValueError("Trial specified as gcisd_complex, but amplitudes.npz not found.")

    else:
        # Try to load trial from pickle file
        try:
            with open(directory + "/trial.pkl", "rb") as f:
                [trial, trial_wave_data] = pickle.load(f)
            wave_data.update(trial_wave_data)
            log.log_0(f"# Read trial of type {type(trial).__name__} from trial.pkl.")
        except:
            log.log_0("# trial.pkl not found, make sure to construct the trial separately.")
            trial = None

    return trial, wave_data


def set_prop(options: Options) -> Any:
    """
    Set up the propagator for AFQMC calculation.

    Args:
        options: Dictionary of calculation options

    Returns:
        Propagator object configured according to options
    """
    if options.walker_type == "restricted":
        if options.vhs_mixed_precision:
            prop = propagation.propagator_restricted(
                options.dt,
                options.n_walkers,
                n_batch=options.n_batch,
                vhs_real_dtype=jnp.float32,
                vhs_complex_dtype=jnp.complex64,
            )
        else:
            prop = propagation.propagator_restricted(
                options.dt, options.n_walkers, n_batch=options.n_batch
            )

    elif options.walker_type == "unrestricted":
        if options.free_projection:
            prop = propagation.propagator_unrestricted(
                options.dt,
                options.n_walkers,
                10,  # Hard-coded value for free projection
                n_batch=options.n_batch,
            )
        else:
            if options.vhs_mixed_precision:
                prop = propagation.propagator_unrestricted(
                    options.dt,
                    options.n_walkers,
                    n_batch=options.n_batch,
                    vhs_real_dtype=jnp.float32,
                    vhs_complex_dtype=jnp.complex64,
                )
            else:
                prop = propagation.propagator_unrestricted(
                    options.dt,
                    options.n_walkers,
                    n_batch=options.n_batch,
                )
    elif options.walker_type == "generalized":
        prop = propagation.propagator_generalized(
            options.dt, options.n_walkers, n_batch=options.n_batch
        )
    else:
        raise ValueError(f"Invalid walker type {options.walker_type}.")

    return prop


def set_sampler(options: Dict) -> Any:
    """
    Set up the sampler for AFQMC calculation.

    Args:
        options: Dictionary of calculation options

    Returns:
        Sampler object configured according to options
    """
    return sampling.sampler(
        options.n_prop_steps,
        options.n_ene_blocks,
        options.n_sr_blocks,
        options.n_blocks,
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
    log: Logger,
    options: Optional[Options] = None,
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
    assert log.verbose == options.verbose
    observable = read_observable(norb, options, directory)
    ham, ham_data = set_ham(norb, h0, h1, chol, options.ene0)
    ham_data = apply_symmetry_mask(ham_data, options)
    mo_coeff = load_mo_coefficients(directory)
    trial, wave_data = set_trial(log, options, mo_coeff, norb, nelec_sp, directory)
    prop = set_prop(options)
    sampler = set_sampler(options)

    ## MPI to get rank value
    #mpi_comm = config.setup_comm_no_print()
    #rank = mpi_comm.COMM_WORLD.Get_rank()

    ## Logger
    #log = Logger(sys.stdout, options.verbose, rank)

    log.log_0(f"# norb: {norb}")
    log.log_0(f"# nelec: {nelec_sp}")
    log.log_0("#")
    for k, v in options.__dict__.items():
        log.log_0(f"# {k}: {v}")
    log.log_0("#")

    return ham_data, ham, prop, trial, wave_data, sampler, observable, options


def run_afqmc_calculation(
    log: Logger,
    mpi_comm: Optional[Any] = None,
    tmp_dir: Optional[str] = None,
    custom_options: Optional[Options] = None,
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
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = setup_afqmc(
        log, options=custom_options, tmp_dir=directory
    )

    assert trial is not None, "Trial wavefunction is None. Cannot run AFQMC."

    # Initialize timing and synchronize MPI processes
    init_time = time.time()
    comm = mpi_comm.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()

    # Run appropriate AFQMC algorithm
    e_afqmc, err_afqmc = 0.0, 0.0
    if options.free_projection:
        driver.fp_afqmc(
            ham_data,
            ham,
            prop,
            trial,
            wave_data,
            sampler,
            observable,
            options,
            mpi_comm,
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
    log.log_0(f"AFQMC total walltime: {end_time - init_time}")
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

    # MPI rank needed for the logger
    mpi_comm = config.setup_comm_no_print()
    rank = mpi_comm.COMM_WORLD.Get_rank()

    # Logger
    log = Logger(sys.stdout, args.verbose, rank)    

    mpi_comm = config.setup_comm(log)
    config.setup_jax(log)
    run_afqmc_calculation(log, mpi_comm=mpi_comm, tmp_dir=args.tmpdir)


if __name__ == "__main__":
    main()
