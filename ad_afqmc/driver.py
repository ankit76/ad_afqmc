import pickle
import time
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes, jvp, random, vjp

from ad_afqmc import (
    hamiltonian,
    misc,
    propagation,
    sampling,
    stat_utils,
    wavefunctions,
    grad_utils,
)
from ad_afqmc.config import mpi_print as print


def afqmc_energy(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
    options: dict,
    MPI: Any,
    init_walkers: Optional[Union[List, jax.Array]] = None,
    tmpdir: str = ".",
) -> Tuple[float, float]:
    """
    Run AFQMC simulation for calculating energy.

    Args:
        ham_data (dict): Hamiltonian data.
        ham (hamiltonian.hamiltonian): Hamiltonian object.
        propagator (propagation.propagator): Propagator object.
        trial (wavefunctions.wave_function): Trial wavefunction.
        wave_data (dict): Wavefunction data.
        sampler (sampling.sampler): Sampler object.
        options (dict): Options for the simulation.
        MPI: MPI object. Either mpi4py object or a dummy handler config.not_MPI.
        init_walkers (Optional[Union[List, jax.Array]], optional): Initial walkers.
        tmpdir (str, optional): Directory for temporary files.

    Returns:
        Tuple[float, float]: AFQMC energy and error
    """
    init = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    seed = options["seed"]

    if rank == 0:
        sha1, branch, local_mods = misc.get_git_info()
        sys_info = misc.print_env_info(sha1, branch, local_mods)

    # Initialize data
    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, propagator, trial, wave_data
    )
    Seed = seed + rank
    prop_data = propagator.init_prop_data(trial, wave_data, ham_data, Seed, init_walkers)
    if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
        raise ValueError(
            "Initial overlaps are zero. Pass walkers with non-zero overlap."
        )
    #prop_data["key"] = random.PRNGKey(seed + rank)

    # Equilibration phase
    comm.Barrier()
    init_time = time.time() - init
    print("# Equilibration sweeps:")
    print(
        f"# {'Iter':>10}      {'Total block weight':<20} {'Block energy':<20} {'Walltime':<10}"
    )
    # print("#   Iter        Block energy      Walltime")
    n = 0
    print(
        f"# {n:>10}      {jnp.sum(prop_data['weights']) * size:<20.9e} {prop_data['e_estimate']:<20.9e} {init_time:<10.2e} "
    )
    # print(f"# {n:5d}      {prop_data['e_estimate']:.9e}     {init_time:.2e} ")
    comm.Barrier()

    n_ene_blocks_eql = options["n_ene_blocks_eql"]
    n_sr_blocks_eql = options["n_sr_blocks_eql"]
    n_eql = options["n_eql"]
    sampler_eq = type(sampler)(
        n_prop_steps=50,
        n_ene_blocks=n_ene_blocks_eql,
        n_sr_blocks=n_sr_blocks_eql,
        n_blocks=n_eql,
    )

    # Run equilibration
    prop_data = _run_equilibration(
        ham,
        ham_data,
        propagator,
        prop_data,
        trial,
        wave_data,
        sampler_eq,
        init,
        MPI,
    )

    # Sampling phase
    comm.Barrier()
    print("#\n# Sampling sweeps:")
    print("#  Iter        Mean energy          Stochastic error       Walltime")
    comm.Barrier()

    global_block_weights = None
    global_block_energies = None
    if rank == 0:
        global_block_weights = np.zeros(sampler.n_blocks)
        global_block_energies = np.zeros(sampler.n_blocks)

    # Run sampling
    for n in range(sampler.n_blocks):
        block_energy_n, prop_data = sampler.propagate_phaseless(
            ham, ham_data, propagator, prop_data, trial, wave_data
        )

        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")

        gather_weights = np.zeros(0, dtype="float32")
        gather_energies = np.zeros(0, dtype="float32")
        if rank == 0:
            gather_weights = np.zeros(size, dtype="float32")
            gather_energies = np.zeros(size, dtype="float32")

        comm.Gather(block_weight_n, gather_weights, root=0)
        comm.Gather(block_energy_n, gather_energies, root=0)
        block_energy_n = 0.0
        if rank == 0:
            global_block_weights[n] = np.sum(gather_weights)
            block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(
                gather_weights
            )
            global_block_energies[n] = block_energy_n

        block_energy_n = comm.bcast(block_energy_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)

        if options["save_walkers"] == True:
            _save_walkers(prop_data, n, tmpdir, rank)

        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

        # Print progress and save intermediate results
        if n % (max(sampler.n_blocks // 10, 1)) == 0:
            _print_progress_energy(
                n,
                global_block_weights,
                global_block_energies,
                rank,
                init,
                comm,
                tmpdir,
            )
            try:
                print(f"node encounters on proc 0: {prop_data['node_crossings']}")
            except:
                pass

    # Analysis phase
    comm.Barrier()
    e_afqmc, e_err_afqmc = _analyze_energy_results(
        global_block_weights, global_block_energies, rank, comm, tmpdir
    )

    return e_afqmc, e_err_afqmc


def afqmc_LNOenergy(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
    options: dict,
    MPI: Any,
    init_walkers: Optional[Union[List, jax.Array]] = None,
    tmpdir: str = ".",
) -> Tuple[float, float]:
    """
    Run AFQMC simulation for calculating energy.

    Args:
        ham_data (dict): Hamiltonian data.
        ham (hamiltonian.hamiltonian): Hamiltonian object.
        propagator (propagation.propagator): Propagator object.
        trial (wavefunctions.wave_function): Trial wavefunction.
        wave_data (dict): Wavefunction data.
        sampler (sampling.sampler): Sampler object.
        options (dict): Options for the simulation.
        MPI: MPI object. Either mpi4py object or a dummy handler config.not_MPI.
        init_walkers (Optional[Union[List, jax.Array]], optional): Initial walkers.
        tmpdir (str, optional): Directory for temporary files.

    Returns:
        Tuple[float, float]: AFQMC energy and error
    """

    init = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    seed = options["seed"]
    truncate = 0  # Message to pass if I need to truncate the n_blocks in all ranks
    truncate_at_n = (
        sampler.n_blocks
    )  # block at which to truncate the n_blocks in all ranks
    maxError = options["maxError"]

    if rank == 0:
        sha1, branch, local_mods = misc.get_git_info()
        sys_info = misc.print_env_info(sha1, branch, local_mods)

    # Initialize data
    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, propagator, trial, wave_data
    )
    local_seed = seed + rank
    prop_data = propagator.init_prop_data(trial, wave_data, ham_data, local_seed, init_walkers)
    if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
        raise ValueError(
            "Initial overlaps are zero. Pass walkers with non-zero overlap."
        )
    # prop_data["key"] = random.PRNGKey(seed + rank)

    # Equilibration phase
    comm.Barrier()
    init_time = time.time() - init
    print("# Equilibration sweeps:")
    print(
        f"# {'Iter':>10}      {'Total block weight':<20} {'Block energy':<20} {'Walltime':<10}"
    )
    # print("#   Iter        Block energy      Walltime")
    n = 0
    print(
        f"# {n:>10}      {jnp.sum(prop_data['weights']) * size:<20.9e} {prop_data['e_estimate']:<20.9e} {init_time:<10.2e} "
    )
    # print(f"# {n:5d}      {prop_data['e_estimate']:.9e}     {init_time:.2e} ")
    comm.Barrier()

    n_ene_blocks_eql = options["n_ene_blocks_eql"]
    n_sr_blocks_eql = options["n_sr_blocks_eql"]
    n_eql = options["n_eql"]
    sampler_eq = sampling.sampler(
        n_prop_steps=50,
        n_ene_blocks=n_ene_blocks_eql,
        n_sr_blocks=n_sr_blocks_eql,
        n_blocks=n_eql,
    )

    # Run equilibration
    prop_data = _run_equilibration(
        ham,
        ham_data,
        propagator,
        prop_data,
        trial,
        wave_data,
        sampler_eq,
        init,
        MPI,
    )

    # Sampling phase
    comm.Barrier()
    print("#\n# Sampling sweeps:")
    print("#  Iter        Mean energy          Stochastic error       Walltime")
    comm.Barrier()

    global_block_weights = None
    global_block_energies = None
    global_block_orbEs = None
    if rank == 0:
        global_block_weights = np.zeros(sampler.n_blocks)
        global_block_energies = np.zeros(sampler.n_blocks)
        global_block_orbEs = np.zeros(sampler.n_blocks)

    # Run sampling
    for n in range(sampler.n_blocks):
        block_energy_n, prop_data, block_orbE_n = sampler.propagate_phaseless(
            ham, ham_data, propagator, prop_data, trial, wave_data
        )

        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        block_orbE_n = np.array([block_orbE_n], dtype="float32")

        gather_weights = np.zeros(0, dtype="float32")
        gather_energies = np.zeros(0, dtype="float32")
        gather_orbE = np.zeros(0, dtype="float32")

        if rank == 0:
            gather_weights = np.zeros(size, dtype="float32")
            gather_energies = np.zeros(size, dtype="float32")
            gather_orbE = np.zeros(size, dtype="float32")

        comm.Gather(block_weight_n, gather_weights, root=0)
        comm.Gather(block_energy_n, gather_energies, root=0)
        comm.Gather(block_orbE_n, gather_orbE, root=0)
        block_energy_n = 0.0
        block_orbE_n = 0.0
        if rank == 0:
            global_block_weights[n] = np.sum(gather_weights)
            block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(
                gather_weights
            )
            block_orbE_n = np.sum(gather_weights * gather_orbE) / np.sum(gather_weights)
            global_block_energies[n] = block_energy_n
            global_block_orbEs[n] = block_orbE_n

        block_energy_n = comm.bcast(block_energy_n, root=0)
        block_orbE_n = comm.bcast(block_orbE_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)

        if options["save_walkers"] == True:
            _save_walkers(prop_data, n, tmpdir, rank)

        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

        # Print progress and save intermediate results
        if n % (max(sampler.n_blocks // 10, 1)) == 0:
            _print_progress_energy(
                n,
                global_block_weights,
                global_block_energies,
                rank,
                init,
                comm,
                tmpdir,
                global_block_orbEs=global_block_orbEs,
            )
            try:
                print(f"node encounters on proc 0: {prop_data['node_crossings']}")
            except:
                pass

        if rank == 0:
            if n % (max(sampler.n_blocks // 5, 1)) == 0 and n > 4:
                orbE_sem = jnp.std(global_block_orbEs[: n + 1]) / jnp.sqrt(n + 1)
                if orbE_sem < maxError:
                    print(
                        f"#\n# Orbital energy convergence achieved: {orbE_sem:.6e} < {maxError:.6e}"
                    )
                    truncate = 1
                    truncate_at_n = n

        truncate = comm.bcast(truncate, root=0)
        truncate_at_n = comm.bcast(truncate_at_n, root=0)
        if truncate == 1:
            break

    # Analysis phase
    comm.Barrier()
    if rank == 0:
        assert global_block_weights is not None
        assert global_block_energies is not None
        e_afqmc, e_err_afqmc = _analyze_LNOenergy_results(
            global_block_weights,
            global_block_energies,
            rank,
            comm,
            tmpdir,
            global_block_orbEs=global_block_orbEs,
            truncate_at_n=truncate_at_n,
        )
    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()

    return e_afqmc, e_err_afqmc


def afqmc_observable(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
    observable: Optional[Tuple],
    options: dict,
    MPI,
    init_walkers: Optional[Union[List, jax.Array]] = None,
    tmpdir: str = ".",
) -> Tuple:
    """
    Run AFQMC simulation with observable calculation using automatic differentiation.

    Returns:
        Tuple[float, float, Any]: AFQMC energy, error, and observable-related data
    """
    if options["ad_mode"] is None:
        raise ValueError("ad_mode must be specified for observable calculation")

    init = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    seed = options["seed"]

    # Set up observable operators
    if observable is not None:
        observable_op = jnp.array(observable[0])
        observable_constant = observable[1]
    else:
        observable_op = jnp.array(ham_data["h1"])
        observable_constant = 0.0

    # Initialize RDM operators based on AD mode
    rdm_op, rdm_2_op, trial_rdm2 = _setup_rdm_operators(
        options["ad_mode"], ham_data, ham, trial, wave_data
    )

    # Initialize data
    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, propagator, trial, wave_data
    )
    Seed = seed + rank
    prop_data = propagator.init_prop_data(trial, wave_data, ham_data, Seed, init_walkers)
    if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
        raise ValueError(
            "Initial overlaps are zero. Pass walkers with non-zero overlap."
        )
    #prop_data["key"] = random.PRNGKey(seed + rank)

    trial_observable = np.sum(trial_rdm1 * observable_op)

    # Equilibration phase
    comm.Barrier()
    init_time = time.time() - init
    print("# Equilibration sweeps:")
    print("#   Iter        Block energy      Walltime")
    n = 0
    print(f"# {n:5d}      {prop_data['e_estimate']:.9e}     {init_time:.2e} ")
    comm.Barrier()

    n_ene_blocks_eql = options["n_ene_blocks_eql"]
    n_sr_blocks_eql = options["n_sr_blocks_eql"]
    neql = options["n_eql"]
    sampler_eq = sampling.sampler(
        n_prop_steps=50,
        n_ene_blocks=n_ene_blocks_eql,
        n_sr_blocks=n_sr_blocks_eql,
        n_blocks=neql,
    )

    # Run equilibration
    prop_data = _run_equilibration(
        ham,
        ham_data,
        propagator,
        prop_data,
        trial,
        wave_data,
        sampler_eq,
        init,
        MPI,
    )

    # Sampling phase
    comm.Barrier()
    print("#\n# Sampling sweeps:")
    print(
        "#  Iter        Mean energy          Stochastic error       Mean observable       Walltime"
    )
    comm.Barrier()

    # Setup for sampling phase
    global_block_weights = None
    global_block_energies = None
    global_block_observables = None
    global_block_rdm1s = None
    global_block_rdm2s = None
    if rank == 0:
        global_block_weights = np.zeros(size * sampler.n_blocks)
        global_block_energies = np.zeros(size * sampler.n_blocks)
        global_block_observables = np.zeros(size * sampler.n_blocks)
        if options["ad_mode"] == "reverse":
            global_block_rdm1s = np.zeros(
                (size * sampler.n_blocks, *(ham_data["h1"].shape))
            )
        elif options["ad_mode"] == "2rdm":
            global_block_rdm2s = np.zeros((size * sampler.n_blocks, *(rdm_2_op.shape)))

    # Set up AD propagation wrapper
    propagate_phaseless_wrapper = _setup_propagate_phaseless_wrapper(
        options, ham, ham_data, propagator, trial, wave_data, sampler
    )

    # Initialize prop_data_tangent and block data
    prop_data_tangent = _init_prop_data_tangent(prop_data)
    block_rdm1_n = np.zeros_like(ham_data["h1"])
    block_rdm2_n = None
    if options["ad_mode"] == "2rdm" or options["ad_mode"] == "nuc_grad":
        block_rdm2_n = np.zeros_like(rdm_2_op)

    # Run sampling with AD
    local_large_deviations = np.array(0)
    for n in range(sampler.n_blocks):
        # Execute appropriate AD mode
        (
            block_energy_n,
            block_observable_n,
            block_rdm1_n,
            block_rdm2_n,
            prop_data,
            local_large_deviations,
        ) = _run_ad_step(
            options["ad_mode"],
            propagate_phaseless_wrapper,
            observable_op,
            observable_constant,
            rdm_op,
            rdm_2_op,
            trial_observable,
            trial_rdm1,
            trial_rdm2,
            prop_data,
            prop_data_tangent,
            block_rdm1_n,
            block_rdm2_n,
            local_large_deviations,
        )

        # Gather results
        (
            block_energy_n,
            global_block_weights,
            global_block_energies,
            global_block_observables,
            global_block_rdm1s,
            global_block_rdm2s,
        ) = _gather_ad_results(
            block_energy_n,
            block_observable_n,
            block_weight_n=np.array([jnp.sum(prop_data["weights"])], dtype="float32"),
            block_rdm1_n=block_rdm1_n,
            block_rdm2_n=block_rdm2_n,
            global_block_weights=global_block_weights,
            global_block_energies=global_block_energies,
            global_block_observables=global_block_observables,
            global_block_rdm1s=global_block_rdm1s,
            global_block_rdm2s=global_block_rdm2s,
            n=n,
            size=size,
            rank=rank,
            comm=comm,
            ad_mode=options["ad_mode"],
        )
        if options["ad_mode"] == "nuc_grad":
            block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
            _save_energy_derivatives(
                block_rdm1_n, block_rdm2_n, block_weight_n, tmpdir, rank
            )
        # Update walkers
        block_energy_n = comm.bcast(block_energy_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)
        if options["save_walkers"] == True:
            _save_walkers(prop_data, n, tmpdir, rank)
        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

        # Print progress and save intermediate results
        if n % (max(sampler.n_blocks // 10, 1)) == 0:
            _print_progress_observable(
                n,
                global_block_weights,
                global_block_energies,
                global_block_observables,
                size,
                rank,
                init,
                comm,
                tmpdir,
                options["ad_mode"],
            )

    # Report large deviations
    global_large_deviations = np.array(0)
    comm.Reduce(
        [local_large_deviations, MPI.INT],
        [global_large_deviations, MPI.INT],
        op=MPI.SUM,
        root=0,
    )
    print(f"#\n# Number of large deviations: {global_large_deviations}", flush=True)

    # Analysis phase
    comm.Barrier()
    result_data = _analyze_observable_results(
        global_block_weights,
        global_block_energies,
        global_block_observables,
        global_block_rdm1s,
        global_block_rdm2s,
        options["ad_mode"],
        ham,
        rank,
        comm,
        tmpdir,
    )

    e_afqmc = result_data.get("e_afqmc")
    e_err_afqmc = result_data.get("e_err_afqmc")
    observable_data = result_data.get("observable_data")

    return e_afqmc, e_err_afqmc, observable_data


def _run_equilibration(
    ham: hamiltonian.hamiltonian,
    ham_data: dict,
    propagator: propagation.propagator,
    prop_data: dict,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler_eq: sampling.sampler,
    init: float,
    MPI: Any,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for n in range(1, sampler_eq.n_blocks + 1):
        block_energy_n, prop_data = sampler_eq.propagate_phaseless(
            ham, ham_data, propagator, prop_data, trial, wave_data
        )
        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        block_weighted_energy_n = np.array(
            [block_energy_n * block_weight_n], dtype="float32"
        )
        total_block_energy_n = np.zeros(1, dtype="float32")
        total_block_weight_n = np.zeros(1, dtype="float32")
        comm.Reduce(
            [block_weighted_energy_n, MPI.FLOAT],
            [total_block_energy_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        comm.Reduce(
            [block_weight_n, MPI.FLOAT],
            [total_block_weight_n, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )
        if rank == 0:
            block_weight_n = total_block_weight_n
            block_energy_n = total_block_energy_n / total_block_weight_n
        comm.Bcast(block_weight_n, root=0)
        comm.Bcast(block_energy_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)
        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = (
            0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n[0]
        ).astype("float64")

        comm.Barrier()
        if n % (max(sampler_eq.n_blocks // 5, 1)) == 0:
            print(
                f"# {n:>10}      {block_weight_n[0]:<20.9e} {block_energy_n[0]:<20.9e} {time.time() - init:<10.2e} ",
                flush=True,
            )
            # print(
            #    f"# {n:5d}      {block_energy_n[0]:.9e}     {time.time() - init:.2e} ",
            #    flush=True,
            # )
        comm.Barrier()
    return prop_data


def _setup_rdm_operators(ad_mode, ham_data, ham, trial, wave_data):
    """Set up RDM operators based on AD mode"""
    rdm_op = 0.0 * jnp.array(ham_data["h1"])  # for reverse mode
    rdm_2_op = None
    trial_rdm2 = None

    if ad_mode == "2rdm":
        nchol = ham_data["chol"].shape[0]
        norb = ham.norb
        eri_full = np.einsum(
            "gj,gl->jl",
            ham_data["chol"].reshape(nchol, -1),
            ham_data["chol"].reshape(nchol, -1),
        )
        rdm_2_op = jnp.array(eri_full).reshape(norb, norb, norb, norb)

        # Set up trial 2-RDM
        trial_rdm1 = trial.get_rdm1(wave_data)
        if isinstance(trial_rdm1, np.ndarray) and len(trial_rdm1.shape) == 2:
            trial_rdm1_spatial = 1.0 * trial_rdm1
        else:
            trial_rdm1_spatial = trial_rdm1[0] + trial_rdm1[1]
        trial_rdm2 = (
            np.einsum("ij,kl->ijkl", trial_rdm1_spatial, trial_rdm1_spatial)
            - np.einsum("ij,kl->iklj", trial_rdm1_spatial, trial_rdm1_spatial) / 2
        )
        trial_rdm2 = trial_rdm2.reshape(ham.norb**2, ham.norb**2)
        trial_rdm2 = jnp.array(trial_rdm2)
    elif ad_mode == "nuc_grad":
        rdm_2_op = jnp.array(ham_data["chol"]).reshape((-1, ham.norb, ham.norb)).copy()

    return rdm_op, rdm_2_op, trial_rdm2


def _setup_propagate_phaseless_wrapper(
    options, ham, ham_data, propagator, trial, wave_data, sampler
):
    """Set up the appropriate propagate_phaseless wrapper function based on options"""
    if options["ad_mode"] == "2rdm":
        return lambda x, y, z: sampler.propagate_phaseless_ad_1(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )
    if (
        options["ad_mode"] == "nuc_grad"
        and (options["orbital_rotation"] == False)
        and (options["do_sr"] == False)
    ):
        return lambda x, y, k, z: sampler.propagate_phaseless_nucgrad_norot_nosr(
            ham, ham_data, x, y, k, propagator, z, trial, wave_data
        )
    elif (
        (options["ad_mode"] == "nuc_grad")
        and (options["orbital_rotation"] == False)
        and (options["do_sr"] == True)
    ):
        return lambda x, y, k, z: sampler.propagate_phaseless_nucgrad_norot(
            ham, ham_data, x, y, k, propagator, z, trial, wave_data
        )
    elif (options["ad_mode"] == "nuc_grad") and (options["orbital_rotation"] == True):
        return lambda x, y, k, z: sampler.propagate_phaseless_nucgrad(
            ham, ham_data, x, y, k, propagator, z, trial, wave_data
        )

    elif options["orbital_rotation"] == False and options["do_sr"] == False:
        return lambda x, y, z: sampler.propagate_phaseless_ad_nosr_norot(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )
    elif options["orbital_rotation"] == False:
        return lambda x, y, z: sampler.propagate_phaseless_ad_norot(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )
    elif options["do_sr"] == False:
        return lambda x, y, z: sampler.propagate_phaseless_ad_nosr(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )
    else:
        return lambda x, y, z: sampler.propagate_phaseless_ad(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )


def _init_prop_data_tangent(prop_data):
    """Initialize tangent data for AD"""
    prop_data_tangent = {}
    for x in prop_data:
        if isinstance(prop_data[x], list):
            prop_data_tangent[x] = [np.zeros_like(y) for y in prop_data[x]]
        elif prop_data[x].dtype == "uint32":
            prop_data_tangent[x] = np.zeros(prop_data[x].shape, dtype=dtypes.float0)
        else:
            prop_data_tangent[x] = np.zeros_like(prop_data[x])
    return prop_data_tangent


def _run_ad_step(
    ad_mode,
    propagate_phaseless_wrapper,
    observable_op,
    observable_constant,
    rdm_op,
    rdm_2_op,
    trial_observable,
    trial_rdm1,
    trial_rdm2,
    prop_data,
    prop_data_tangent,
    block_rdm1_n,
    block_rdm2_n,
    local_large_deviations,
):
    block_observable_n = 0.0

    if ad_mode == "forward":
        coupling = 0.0
        block_energy_n, block_observable_n, prop_data = jvp(
            propagate_phaseless_wrapper,
            (coupling, observable_op, prop_data),
            (1.0, 0.0 * observable_op, prop_data_tangent),
            has_aux=True,
        )
        if np.isnan(block_observable_n) or np.isinf(block_observable_n):
            block_observable_n = trial_observable
            local_large_deviations += 1

    elif ad_mode == "reverse":
        coupling = 1.0
        block_energy_n, block_vjp_fun, prop_data = vjp(
            propagate_phaseless_wrapper, coupling, rdm_op, prop_data, has_aux=True
        )
        block_rdm1_n = block_vjp_fun(1.0)[1]
        block_observable_n = np.sum(block_rdm1_n * observable_op)
        if np.isnan(block_observable_n) or np.isinf(block_observable_n):
            block_observable_n = trial_observable
            block_rdm1_n = trial_rdm1
            local_large_deviations += 1

    elif ad_mode == "2rdm":
        coupling = 1.0
        block_energy_n, block_vjp_fun, prop_data = vjp(
            propagate_phaseless_wrapper, coupling, rdm_2_op, prop_data, has_aux=True
        )
        block_rdm2_n = block_vjp_fun(1.0)[1]
        block_observable_n = trial_observable
        if np.isnan(np.linalg.norm(block_rdm2_n)) or np.isinf(
            np.linalg.norm(block_rdm2_n)
        ):
            block_observable_n = trial_observable
            block_rdm2_n = trial_rdm2
            local_large_deviations += 1

    elif ad_mode == "nuc_grad":
        coupling = 1.0
        block_energy_n, block_vjp_fun, prop_data = vjp(
            propagate_phaseless_wrapper,
            coupling,
            rdm_op,
            rdm_2_op,
            prop_data,
            has_aux=True,
        )
        block_rdm1_n = block_vjp_fun(1.0)[1]
        block_rdm2_n = block_vjp_fun(1.0)[2]
        block_observable_n = 0.0

    # Add observable constant if needed
    block_observable_n = block_observable_n + observable_constant

    return (
        block_energy_n,
        block_observable_n,
        block_rdm1_n,
        block_rdm2_n,
        prop_data,
        local_large_deviations,
    )


def _gather_ad_results(
    block_energy_n,
    block_observable_n,
    block_weight_n,
    block_rdm1_n,
    block_rdm2_n,
    global_block_weights,
    global_block_energies,
    global_block_observables,
    global_block_rdm1s,
    global_block_rdm2s,
    n,
    size,
    rank,
    comm,
    ad_mode,
):
    """Gather AD results from all processes"""
    block_energy_n = np.array([block_energy_n], dtype="float32")
    block_observable_n = np.array([block_observable_n], dtype="float32")
    block_rdm1_n = np.array(block_rdm1_n, dtype="float32")
    if ad_mode == "2rdm":
        block_rdm2_n = np.array(block_rdm2_n, dtype="float32")

    gather_weights = np.zeros(0, dtype="float32")
    gather_energies = np.zeros(0, dtype="float32")
    gather_observables = None
    gather_rdm1s = None
    gather_rdm2s = None
    if rank == 0:
        gather_weights = np.zeros(size, dtype="float32")
        gather_energies = np.zeros(size, dtype="float32")
        gather_observables = np.zeros(size, dtype="float32")
        if ad_mode == "reverse":
            gather_rdm1s = np.zeros((size, *block_rdm1_n.shape), dtype="float32")
        elif ad_mode == "2rdm":
            gather_rdm2s = np.zeros((size, *block_rdm2_n.shape), dtype="float32")

    comm.Gather(block_weight_n, gather_weights, root=0)
    comm.Gather(block_energy_n, gather_energies, root=0)
    comm.Gather(block_observable_n, gather_observables, root=0)
    if ad_mode == "reverse":
        comm.Gather(block_rdm1_n, gather_rdm1s, root=0)
    elif ad_mode == "2rdm":
        comm.Gather(block_rdm2_n, gather_rdm2s, root=0)

    if rank == 0:
        global_block_weights[n * size : (n + 1) * size] = gather_weights
        global_block_energies[n * size : (n + 1) * size] = gather_energies
        global_block_observables[n * size : (n + 1) * size] = gather_observables
        if ad_mode == "reverse":
            global_block_rdm1s[n * size : (n + 1) * size] = gather_rdm1s
        elif ad_mode == "2rdm":
            global_block_rdm2s[n * size : (n + 1) * size] = gather_rdm2s
        block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(
            gather_weights
        )

    return (
        block_energy_n,
        global_block_weights,
        global_block_energies,
        global_block_observables,
        global_block_rdm1s,
        global_block_rdm2s,
    )


def _save_energy_derivatives(block_rdm1_n, block_rdm2_n, block_weight_n, tmpdir, rank):
    """Save energy derivatives to file"""
    grad_utils.append_to_array(
        tmpdir + f"/en_der_afqmc_{rank}.npz",
        block_rdm1_n,
        block_rdm2_n,
        block_weight_n,
    )


def _save_walkers(prop_data, n, tmpdir, rank):
    """Save walker data to file"""
    if n > 0:
        with open(f"{tmpdir}/prop_data_{rank}.bin", "ab") as f:
            pickle.dump(prop_data, f)
    else:
        with open(f"{tmpdir}/prop_data_{rank}.bin", "wb") as f:
            pickle.dump(prop_data, f)


def _print_progress_energy(
    n,
    global_block_weights,
    global_block_energies,
    rank,
    init,
    comm,
    tmpdir,
    global_block_orbEs=None,
):
    """Print progress information for energy calculations"""
    comm.Barrier()
    if rank == 0:
        e_afqmc, energy_error = stat_utils.blocking_analysis(
            global_block_weights[: (n + 1)],
            global_block_energies[: (n + 1)],
            neql=0,
        )
        if global_block_orbEs is not None:
            orbE_avg, orbE_error = stat_utils.blocking_analysis(
                global_block_weights[: (n + 1)],
                global_block_orbEs[: (n + 1)],
                neql=0,
            )
            if energy_error is not None and orbE_error is not None:
                print(
                    f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {orbE_avg:.9e}        {orbE_error:.9e}      {time.time() - init:.2e} ",
                    flush=True,
                )
            else:
                print(
                    f" {n:5d}      {e_afqmc:.9e}                -              {orbE_avg:.9e}                -              {time.time() - init:.2e} ",
                    flush=True,
                )

        elif energy_error is not None:
            print(
                f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {time.time() - init:.2e} ",
                flush=True,
            )
        else:
            print(
                f" {n:5d}      {e_afqmc:.9e}                -              {time.time() - init:.2e} ",
                flush=True,
            )
        np.savetxt(
            tmpdir + "/samples_raw.dat",
            np.stack(
                (
                    global_block_weights[: (n + 1)],
                    global_block_energies[: (n + 1)],
                )
            ).T,
        )
    comm.Barrier()


def _print_progress_observable(
    n,
    global_block_weights,
    global_block_energies,
    global_block_observables,
    size,
    rank,
    init,
    comm,
    tmpdir,
    ad_mode,
):
    """Print progress information for observable calculations"""
    comm.Barrier()
    if rank == 0:
        e_afqmc, energy_error = stat_utils.blocking_analysis(
            global_block_weights[: (n + 1) * size],
            global_block_energies[: (n + 1) * size],
            neql=0,
        )
        obs_afqmc, _ = stat_utils.blocking_analysis(
            global_block_weights[: (n + 1) * size],
            global_block_observables[: (n + 1) * size],
            neql=0,
        )
        if energy_error is not None:
            print(
                f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {obs_afqmc:.9e}       {time.time() - init:.2e} ",
                flush=True,
            )
        else:
            print(
                f" {n:5d}      {e_afqmc:.9e}                -              {obs_afqmc:.9e}       {time.time() - init:.2e} ",
                flush=True,
            )
        np.savetxt(
            tmpdir + "/samples_raw.dat",
            np.stack(
                (
                    global_block_weights[: (n + 1) * size],
                    global_block_energies[: (n + 1) * size],
                    global_block_observables[: (n + 1) * size],
                )
            ).T,
        )
    comm.Barrier()


def _analyze_energy_results(
    global_block_weights, global_block_energies, rank, comm, tmpdir
):
    """Analyze energy results and calculate statistics"""
    e_afqmc, e_err_afqmc = None, None
    if rank == 0:
        np.savetxt(
            tmpdir + "/samples_raw.dat",
            np.stack((global_block_weights, global_block_energies)).T,
        )

        # Clean up outliers
        samples_clean, _ = stat_utils.reject_outliers(
            np.stack((global_block_weights, global_block_energies)).T, 1
        )
        print(
            f"# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} "
        )
        np.savetxt(tmpdir + "/samples.dat", samples_clean)

        clean_weights = samples_clean[:, 0]
        clean_energies = samples_clean[:, 1]

        # Calculate final statistics
        e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
            clean_weights, clean_energies, neql=0, printQ=True
        )

        # Print formatted results
        if e_err_afqmc is not None:
            sig_dec = int(abs(np.floor(np.log10(e_err_afqmc))))
            sig_err = np.around(
                np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
            )
            sig_e = np.around(e_afqmc, sig_dec)
            print(f"AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
        elif e_afqmc is not None:
            print(f"Could not determine stochastic error automatically\n", flush=True)
            print(f"AFQMC energy: {e_afqmc}\n", flush=True)
            e_err_afqmc = 0.0

    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()

    return e_afqmc, e_err_afqmc


def _analyze_observable_results(
    global_block_weights,
    global_block_energies,
    global_block_observables,
    global_block_rdm1s,
    global_block_rdm2s,
    ad_mode,
    ham,
    rank,
    comm,
    tmpdir,
):
    """Analyze observable results and calculate statistics"""
    result_data = {}

    if rank == 0:
        # Save raw samples
        np.savetxt(
            tmpdir + "/samples_raw.dat",
            np.stack(
                (global_block_weights, global_block_energies, global_block_observables)
            ).T,
        )

        # Clean up outliers based on AD mode
        if ad_mode != "2rdm":
            samples_clean, idx = stat_utils.reject_outliers(
                np.stack(
                    (
                        global_block_weights,
                        global_block_energies,
                        global_block_observables,
                    )
                ).T,
                2,
            )
        else:
            samples_clean, idx = stat_utils.reject_outliers(
                np.stack(
                    (
                        global_block_weights,
                        global_block_energies,
                        global_block_observables,
                    )
                ).T,
                1,
            )

        print(
            f"# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} "
        )
        np.savetxt(tmpdir + "/samples.dat", samples_clean)

        clean_weights = samples_clean[:, 0]
        clean_energies = samples_clean[:, 1]
        clean_observables = samples_clean[:, 2]

        if ad_mode == "reverse":
            clean_rdm1s = (
                global_block_rdm1s[idx] if global_block_rdm1s is not None else None
            )
        elif ad_mode == "2rdm":
            clean_rdm2s = (
                global_block_rdm2s[idx] if global_block_rdm2s is not None else None
            )

        # Calculate energy statistics
        e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
            clean_weights, clean_energies, neql=0, printQ=True
        )

        # Print formatted energy results
        if e_err_afqmc is not None:
            sig_dec = int(abs(np.floor(np.log10(e_err_afqmc))))
            sig_err = np.around(
                np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
            )
            sig_e = np.around(e_afqmc, sig_dec)
            print(f"AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
        elif e_afqmc is not None:
            print(f"AFQMC energy: {e_afqmc}\n", flush=True)
            e_err_afqmc = 0.0

        # Calculate observable statistics
        obs_afqmc, obs_err_afqmc = stat_utils.blocking_analysis(
            clean_weights, clean_observables, neql=0, printQ=True
        )

        # Print formatted observable results
        if obs_err_afqmc is not None:
            sig_dec = int(abs(np.floor(np.log10(obs_err_afqmc))))
            sig_err = np.around(
                np.round(obs_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
            )
            sig_obs = np.around(obs_afqmc, sig_dec)
            print(
                f"AFQMC observable: {sig_obs:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n"
            )
        elif obs_afqmc is not None:
            print(f"AFQMC observable: {obs_afqmc}\n", flush=True)
        else:
            obs_afqmc = 0.0
            obs_err_afqmc = 0.0

        observable_data = {"obs_afqmc": obs_afqmc, "obs_err_afqmc": obs_err_afqmc}
        obs_err_afqmc = obs_err_afqmc if obs_err_afqmc is not None else np.nan
        np.savetxt(tmpdir + "/obs_err.txt", np.array([obs_afqmc, obs_err_afqmc]))

        # Additional analysis based on AD mode
        if ad_mode == "reverse" and clean_rdm1s is not None:
            # Calculate average RDM1
            norms_rdm1 = np.array(list(map(np.linalg.norm, clean_rdm1s)))
            samples_clean_rdm, idx_rdm = stat_utils.reject_outliers(
                np.stack((clean_weights, norms_rdm1)).T, 1
            )
            clean_weights_rdm = samples_clean_rdm[:, 0]
            clean_rdm1s = clean_rdm1s[idx_rdm]

            avg_rdm1 = np.einsum(
                "i,i...->...", clean_weights_rdm, clean_rdm1s
            ) / np.sum(clean_weights_rdm)
            errors_rdm1 = np.array(
                list(map(np.linalg.norm, clean_rdm1s - avg_rdm1))
            ) / np.linalg.norm(avg_rdm1)

            print(f"# RDM noise:", flush=True)
            rdm_noise, rdm_noise_err = stat_utils.blocking_analysis(
                clean_weights_rdm, errors_rdm1, neql=0, printQ=True
            )

            # Save RDM1 data
            np.savez("rdm1_afqmc.npz", rdm1=avg_rdm1)
            observable_data["rdm1"] = avg_rdm1
            observable_data["rdm1_noise"] = rdm_noise
            observable_data["rdm1_noise_err"] = rdm_noise_err

        elif ad_mode == "2rdm" and clean_rdm2s is not None:
            # Calculate average RDM2
            norms_rdm2 = np.array(list(map(np.linalg.norm, clean_rdm2s)))
            samples_clean_rdm, idx_rdm = stat_utils.reject_outliers(
                np.stack((clean_weights, norms_rdm2)).T, 1
            )
            clean_weights_rdm = samples_clean_rdm[:, 0]
            clean_rdm2s = clean_rdm2s[idx_rdm]

            avg_rdm2 = np.einsum(
                "i,i...->...", clean_weights_rdm, clean_rdm2s
            ) / np.sum(clean_weights_rdm)
            errors_rdm2 = np.array(
                list(map(np.linalg.norm, clean_rdm2s - avg_rdm2))
            ) / np.linalg.norm(avg_rdm2)

            print(f"# 2RDM noise:", flush=True)
            rdm2_noise, rdm2_noise_err = stat_utils.blocking_analysis(
                clean_weights_rdm, errors_rdm2, neql=0, printQ=True
            )

            # Save RDM2 data
            np.savez(
                "rdm2_afqmc.npz",
                rdm2=2 * avg_rdm2.reshape(ham.norb, ham.norb, ham.norb, ham.norb),
            )
            observable_data["rdm2"] = 2 * avg_rdm2.reshape(
                ham.norb, ham.norb, ham.norb, ham.norb
            )
            observable_data["rdm2_noise"] = rdm2_noise
            observable_data["rdm2_noise_err"] = rdm2_noise_err
        elif ad_mode == "nuc_grad":
            grad_utils.calculate_nuc_gradients(tmpdir=tmpdir)

        result_data = {
            "e_afqmc": e_afqmc,
            "e_err_afqmc": e_err_afqmc,
            "observable_data": observable_data,
        }

    comm.Barrier()
    # Broadcast results (simplified for this example)
    e_afqmc = comm.bcast(result_data.get("e_afqmc") if rank == 0 else None, root=0)
    e_err_afqmc = comm.bcast(
        result_data.get("e_err_afqmc") if rank == 0 else None, root=0
    )

    if rank == 0:
        return result_data
    else:
        return {"e_afqmc": e_afqmc, "e_err_afqmc": e_err_afqmc, "observable_data": None}


def _analyze_LNOenergy_results(
    global_block_weights: np.ndarray,
    global_block_energies: np.ndarray,
    rank: int,
    comm,
    tmpdir: str,
    global_block_orbEs: Optional[np.ndarray] = None,
    truncate_at_n: Optional[int] = None,
):
    """Analyze energy results and calculate statistics"""
    e_afqmc, e_err_afqmc = None, None
    if rank == 0:
        if truncate_at_n is not None:
            global_block_weights = global_block_weights[: truncate_at_n + 1]
            global_block_energies = global_block_energies[: truncate_at_n + 1]
            global_block_orbEs = global_block_orbEs[: truncate_at_n + 1]
        np.savetxt(
            tmpdir + "/samples_raw.dat",
            np.stack((global_block_weights, global_block_energies)).T,
        )

        # Clean up outliers
        samples_clean, _ = stat_utils.reject_outliers(
            np.stack((global_block_weights, global_block_energies)).T, 1
        )
        print(
            f"# Number of outliers in post: {global_block_weights.size - samples_clean.shape[0]} "
        )
        np.savetxt(tmpdir + "/samples.dat", samples_clean)

        clean_weights = samples_clean[:, 0]
        clean_energies = samples_clean[:, 1]

        # Calculate final statistics
        e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
            clean_weights, clean_energies, neql=0, printQ=True
        )
        orbE_afqmc, orbE_err_afqmc = stat_utils.blocking_analysis(
            global_block_weights, global_block_orbEs, neql=0, printQ=True
        )
        if orbE_err_afqmc is not None:
            print(
                f"Orbital energy: {orbE_afqmc:.9e} +/- {orbE_err_afqmc:.9e}\n",
                flush=True,
            )
        else:
            print(f"Could not determine orbital energy automatically\n", flush=True)
            print(f"Orbital energy: {orbE_afqmc}\n", flush=True)
            orbE_err_afqmc = 0.0

        # Print formatted results
        if e_err_afqmc is not None:
            sig_dec = int(abs(np.floor(np.log10(e_err_afqmc))))
            sig_err = np.around(
                np.round(e_err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
            )
            sig_e = np.around(e_afqmc, sig_dec)
            print(f"AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n")
        elif e_afqmc is not None:
            print(f"Could not determine stochastic error automatically\n", flush=True)
            print(f"AFQMC energy: {e_afqmc}\n", flush=True)
            e_err_afqmc = 0.0

    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()

    return e_afqmc, e_err_afqmc


# Keep the original function for backward compatibility
def afqmc(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
    observable,
    options: dict,
    MPI,
    init_walkers: Optional[Union[List, jax.Array]] = None,
    tmpdir: str = ".",
):
    """
    Legacy function for backward compatibility. Calls either afqmc_energy or afqmc_observable
    based on options["ad_mode"].
    """
    if options["ad_mode"] is None:
        if options["prjlo"] is not None:
            return afqmc_LNOenergy(
                ham_data=ham_data,
                ham=ham,
                propagator=propagator,
                trial=trial,
                wave_data=wave_data,
                sampler=sampler,
                options=options,
                MPI=MPI,
                init_walkers=init_walkers,
                tmpdir=tmpdir,
            )

        else:
            return afqmc_energy(
                ham_data=ham_data,
                ham=ham,
                propagator=propagator,
                trial=trial,
                wave_data=wave_data,
                sampler=sampler,
                options=options,
                MPI=MPI,
                init_walkers=init_walkers,
                tmpdir=tmpdir,
            )
    else:
        e_afqmc, e_err_afqmc, _ = afqmc_observable(
            ham_data=ham_data,
            ham=ham,
            propagator=propagator,
            trial=trial,
            wave_data=wave_data,
            sampler=sampler,
            observable=observable,
            options=options,
            MPI=MPI,
            init_walkers=init_walkers,
            tmpdir=tmpdir,
        )
        return e_afqmc, e_err_afqmc


def fp_afqmc(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    trial_ket: wavefunctions.wave_function,
    wave_data_ket: dict,
    sampler: sampling.sampler,
    observable,
    options: dict,
    MPI,
    init_walkers: Optional[Union[List, jax.Array]] = None,
):
    init = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    seed = options["seed"]

    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, propagator, trial, wave_data
    )
    Seed = seed + rank
    prop_data = propagator.init_prop_data(trial_ket, wave_data_ket, ham_data, Seed, init_walkers, trial, wave_data)
    #prop_data["key"] = random.PRNGKey(seed + rank)
    
    comm.Barrier()
    init_time = time.time() - init
    print("#\n# Sampling sweeps:")
    print("#  Iter        Mean energy          Stochastic error       Walltime")
    comm.Barrier()

    # global_block_weights = np.zeros(size * sampler.n_ene_blocks) + 0.0j
    # global_block_energies = np.zeros(size * sampler.n_ene_blocks) + 0.0j

    total_energy = np.zeros((sampler.n_ene_blocks, sampler.n_blocks+1)) + 0.0j
    total_weight = np.zeros((sampler.n_ene_blocks, sampler.n_blocks+1)) + 0.0j
    total_sign   = np.ones((sampler.n_ene_blocks, sampler.n_blocks+1)) + 0.0j

    avg_energy = np.zeros((sampler.n_blocks)) + 0.0j
    avg_weight = np.zeros((sampler.n_blocks)) + 0.0j
    for n in range(
        sampler.n_ene_blocks
    ):  # hacking this variable for number of trajectories

        ##initialize a new set of determinants every block
        ##if the ket is CCSD that is being sampled then good to sample it many times
        if (n != 0):
            prop_data["walkers"], prop_data = trial_ket.get_init_walkers(
                wave_data_ket, propagator.n_walkers, "unrestricted" if isinstance(prop_data["walkers"], list) else "restricted", prop_data
            )

            energy_samples = jnp.real(
                trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
            )
            e_estimate = jnp.array(jnp.sum(energy_samples) / propagator.n_walkers)
            prop_data["e_estimate"] = e_estimate

        total_sign[n,0] = jnp.sum(prop_data["overlaps"])/jnp.sum(jnp.abs(prop_data["overlaps"]))
        total_energy[n,0] = prop_data["e_estimate"]
        total_weight[n,0] = jnp.sum(prop_data["weights"])

        (
            prop_data_tr,
            energy_samples,
            weights,
            prop_data["key"],
        ) = sampler.propagate_free(
            ham, ham_data, propagator, prop_data, trial, wave_data
        )
        # global_block_weights[n] = weights[0]
        # global_block_energies[n] = energy_samples[0]
        avg_sign = jax.vmap(lambda ov : jnp.sum(ov)/jnp.sum(jnp.abs(ov)))(prop_data_tr["overlaps"])
        total_energy[n,1:] = energy_samples
        total_weight[n,1:] = weights
        total_sign[n,1:] = avg_sign

        avg_weight += weights
        avg_energy += weights * (energy_samples - avg_energy) / avg_weight
        if options["save_walkers"] == True:
            if n > 0:
                with open(f"prop_data_{rank}.bin", "ab") as f:
                    pickle.dump(prop_data_tr, f)
            else:
                with open(f"prop_data_{rank}.bin", "wb") as f:
                    pickle.dump(prop_data_tr, f)

        #if n % (max(sampler.n_ene_blocks // 10, 1)) == 0:
        comm.Barrier()
        if rank == 0:
            for i in range(avg_energy.shape[0]):
                print("{0:5.3f}  {1:18.9f}  {2:8.2f}".format((i+1)*propagator.dt * sampler.n_prop_steps, avg_energy[i].real, avg_sign[i].real))
            print("")
        times = propagator.dt * sampler.n_prop_steps * jnp.arange(sampler.n_blocks+1)
        mean_energies = np.sum(total_energy[:n+1] * total_weight[:n+1], axis=0) / np.sum(total_weight[:n+1], axis=0)
        error = np.std(total_energy[:n+1], axis=0) / (n)**0.5
        np.savetxt(
            "samples_raw.dat", np.stack((times, mean_energies.real, error.real, np.mean(total_sign[:n+1], axis=0).real)).T
        )
        np.savetxt(
            "RawEnergies.dat", total_energy[:n+1].T.real
        )
        #print(f"{n:5d}: {mean_energies.real}")
