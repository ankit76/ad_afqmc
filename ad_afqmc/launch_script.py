import argparse
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ad_afqmc import config, driver, utils
from ad_afqmc.config import mpi_print as print

# This file is used for launching MPI processes for AFQMC calculations.

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
    ham_data, ham, prop, trial, wave_data, sampler, observable, options = (
        utils.setup_afqmc(options=custom_options, tmp_dir=directory)
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
