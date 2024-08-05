import os
import time

import numpy as np

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import pickle
from copy import deepcopy
from functools import partial
from typing import Optional, Sequence

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
from jax import dtypes, jvp, random, vjp
from mpi4py import MPI

from ad_afqmc import hamiltonian, propagation, sampling, stat_utils, wavefunctions

print = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def afqmc(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
    observable,
    options: dict,
    init_walkers: Optional[Sequence] = None,
):
    init = time.time()
    seed = options["seed"]
    neql = options["n_eql"]

    if observable is not None:
        observable_op = jnp.array(observable[0])
        observable_constant = observable[1]
    else:
        observable_op = jnp.array(ham_data["h1"])
        observable_constant = 0.0

    rdm_op = 0.0 * jnp.array(ham_data["h1"])  # for reverse mode
    rdm_2_op = None
    if options["ad_mode"] == "2rdm":
        nchol = ham.nchol
        norb = ham.norb
        eri_full = np.einsum(
            "gj,gl->jl",
            ham_data["chol"].reshape(nchol, -1),
            ham_data["chol"].reshape(nchol, -1),
        )
        rdm_2_op = jnp.array(eri_full).reshape(norb, norb, norb, norb)

    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, propagator, trial, wave_data
    )
    prop_data = propagator.init_prop_data(trial, wave_data, ham_data, init_walkers)
    if jnp.abs(jnp.sum(prop_data["overlaps"])) < 1.0e-6:
        raise ValueError(
            "Initial overlaps are zero. Pass walkers with non-zero overlap."
        )
    prop_data["key"] = random.PRNGKey(seed + rank)

    trial_observable = np.sum(trial_rdm1 * observable_op)
    trial_rdm2 = None
    if options["ad_mode"] == "2rdm":
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

    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("# Equilibration sweeps:")
        print("#   Iter        Block energy      Walltime")
        n = 0
        print(f"# {n:5d}      {prop_data['e_estimate']:.9e}     {init_time:.2e} ")
    comm.Barrier()

    # propagator_eq = deepcopy(propagator)
    sampler_eq = sampling.sampler(n_prop_steps=50, n_ene_blocks=5, n_sr_blocks=10)

    for n in range(1, neql + 1):
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
        )

        comm.Barrier()
        if rank == 0:
            print(
                f"# {n:5d}      {block_energy_n[0]:.9e}     {time.time() - init:.2e} ",
                flush=True,
            )
        comm.Barrier()

    local_large_deviations = np.array(0)

    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("#\n# Sampling sweeps:")
        if options["ad_mode"] is None:
            print("#  Iter        Mean energy          Stochastic error       Walltime")
        else:
            print(
                "#  Iter        Mean energy          Stochastic error       Mean observable       Walltime"
            )
    comm.Barrier()

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

    if options["orbital_rotation"] == False and options["do_sr"] == False:
        propagate_phaseless_wrapper = (
            lambda x, y, z: sampler.propagate_phaseless_ad_nosr_norot(
                ham, ham_data, x, y, propagator, z, trial, wave_data
            )
        )
    elif options["orbital_rotation"] == False:
        propagate_phaseless_wrapper = (
            lambda x, y, z: sampler.propagate_phaseless_ad_norot(
                ham, ham_data, x, y, propagator, z, trial, wave_data
            )
        )
    elif options["do_sr"] == False:
        propagate_phaseless_wrapper = (
            lambda x, y, z: sampler.propagate_phaseless_ad_nosr(
                ham, ham_data, x, y, propagator, z, trial, wave_data
            )
        )
    else:
        propagate_phaseless_wrapper = lambda x, y, z: sampler.propagate_phaseless_ad(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )

    if options["ad_mode"] == "2rdm":
        propagate_phaseless_wrapper = lambda x, y, z: sampler.propagate_phaseless_ad_1(
            ham, ham_data, x, y, propagator, z, trial, wave_data
        )

    prop_data_tangent = {}
    for x in prop_data:
        if isinstance(prop_data[x], list):
            prop_data_tangent[x] = [np.zeros_like(y) for y in prop_data[x]]
        elif prop_data[x].dtype == "uint32":
            prop_data_tangent[x] = np.zeros(prop_data[x].shape, dtype=dtypes.float0)
        else:
            prop_data_tangent[x] = np.zeros_like(prop_data[x])
    block_rdm1_n = np.zeros_like(ham_data["h1"])
    block_rdm2_n = None
    if options["ad_mode"] == "2rdm":
        block_rdm2_n = np.zeros_like(rdm_2_op)
    block_observable_n = 0.0

    for n in range(sampler.n_blocks):
        if options["ad_mode"] == "forward":
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
        elif options["ad_mode"] == "reverse":
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
        elif options["ad_mode"] == "2rdm":
            coupling = 1.0
            block_energy_n, block_vjp_fun, prop_data = vjp(
                propagate_phaseless_wrapper, coupling, rdm_2_op, prop_data, has_aux=True
            )
            block_rdm2_n = block_vjp_fun(1.0)[1]
            block_observable_n = trial_observable
            # if np.isnan(block_observable_n) or np.isinf(block_observable_n) :
            if np.isnan(np.linalg.norm(block_rdm2_n)) or np.isinf(
                np.linalg.norm(block_rdm2_n)
            ):
                block_observable_n = trial_observable
                block_rdm2_n = trial_rdm2
                local_large_deviations += 1
        else:
            block_energy_n, prop_data = sampler.propagate_phaseless(
                ham, ham_data, propagator, prop_data, trial, wave_data
            )
            block_observable_n = 0.0

        block_energy_n = np.array([block_energy_n], dtype="float32")
        block_observable_n = np.array(
            [block_observable_n + observable_constant], dtype="float32"
        )
        block_weight_n = np.array([jnp.sum(prop_data["weights"])], dtype="float32")
        block_rdm1_n = np.array(block_rdm1_n, dtype="float32")
        if options["ad_mode"] == "2rdm":
            block_rdm2_n = np.array(block_rdm2_n, dtype="float32")

        gather_weights = None
        gather_energies = None
        gather_observables = None
        gather_rdm1s = None
        gather_rdm2s = None
        if rank == 0:
            gather_weights = np.zeros(size, dtype="float32")
            gather_energies = np.zeros(size, dtype="float32")
            gather_observables = np.zeros(size, dtype="float32")
            if options["ad_mode"] == "reverse":
                gather_rdm1s = np.zeros(
                    (size, *(ham_data["h1"].shape)), dtype="float32"
                )
            elif options["ad_mode"] == "2rdm":
                gather_rdm2s = np.zeros((size, *(rdm_2_op.shape)), dtype="float32")

        comm.Gather(block_weight_n, gather_weights, root=0)
        comm.Gather(block_energy_n, gather_energies, root=0)
        comm.Gather(block_observable_n, gather_observables, root=0)
        if options["ad_mode"] == "reverse":
            comm.Gather(block_rdm1_n, gather_rdm1s, root=0)
        elif options["ad_mode"] == "2rdm":
            comm.Gather(block_rdm2_n, gather_rdm2s, root=0)
        block_energy_n = 0.0
        if rank == 0:
            global_block_weights[n * size : (n + 1) * size] = gather_weights
            global_block_energies[n * size : (n + 1) * size] = gather_energies
            global_block_observables[n * size : (n + 1) * size] = gather_observables
            if options["ad_mode"] == "reverse":
                global_block_rdm1s[n * size : (n + 1) * size] = gather_rdm1s
            elif options["ad_mode"] == "2rdm":
                global_block_rdm2s[n * size : (n + 1) * size] = gather_rdm2s
            block_energy_n = np.sum(gather_weights * gather_energies) / np.sum(
                gather_weights
            )

        block_energy_n = comm.bcast(block_energy_n, root=0)
        prop_data = propagator.orthonormalize_walkers(prop_data)

        if options["save_walkers"] == True:
            if n > 0:
                with open(f"prop_data_{rank}.bin", "ab") as f:
                    pickle.dump(prop_data, f)
            else:
                with open(f"prop_data_{rank}.bin", "wb") as f:
                    pickle.dump(prop_data, f)

        prop_data = propagator.stochastic_reconfiguration_global(prop_data, comm)
        prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy_n

        if n % (max(sampler.n_blocks // 10, 1)) == 0:
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
                    if options["ad_mode"] is None:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {time.time() - init:.2e} ",
                            flush=True,
                        )
                    else:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {obs_afqmc:.9e}       {time.time() - init:.2e} ",
                            flush=True,
                        )
                else:
                    if options["ad_mode"] is None:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}                -              {time.time() - init:.2e} ",
                            flush=True,
                        )
                    else:
                        print(
                            f" {n:5d}      {e_afqmc:.9e}                -              {obs_afqmc:.9e}       {time.time() - init:.2e} ",
                            flush=True,
                        )
                np.savetxt(
                    "samples_raw.dat",
                    np.stack(
                        (
                            global_block_weights[: (n + 1) * size],
                            global_block_energies[: (n + 1) * size],
                            global_block_observables[: (n + 1) * size],
                        )
                    ).T,
                )
            comm.Barrier()

    global_large_deviations = np.array(0)
    comm.Reduce(
        [local_large_deviations, MPI.INT],
        [global_large_deviations, MPI.INT],
        op=MPI.SUM,
        root=0,
    )
    comm.Barrier()
    if rank == 0:
        print(f"#\n# Number of large deviations: {global_large_deviations}", flush=True)

    comm.Barrier()
    e_afqmc, e_err_afqmc = None, None
    if rank == 0:
        np.savetxt(
            "samples_raw.dat",
            np.stack(
                (global_block_weights, global_block_energies, global_block_observables)
            ).T,
        )
        if options["ad_mode"] is not None and options["ad_mode"] != "2rdm":
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
        np.savetxt("samples.dat", samples_clean)
        global_block_weights = samples_clean[:, 0]
        global_block_energies = samples_clean[:, 1]
        global_block_observables = samples_clean[:, 2]
        if options["ad_mode"] == "reverse":
            global_block_rdm1s = global_block_rdm1s[idx]
        elif options["ad_mode"] == "2rdm":
            global_block_rdm2s = global_block_rdm2s[idx]

        e_afqmc, e_err_afqmc = stat_utils.blocking_analysis(
            global_block_weights, global_block_energies, neql=0, printQ=True
        )
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

        if options["ad_mode"] is not None:
            obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                global_block_weights, global_block_observables, neql=0, printQ=True
            )
            if err_afqmc is not None:
                sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
                sig_err = np.around(
                    np.round(err_afqmc * 10**sig_dec) * 10 ** (-sig_dec), sig_dec
                )
                sig_obs = np.around(obs_afqmc, sig_dec)
                print(
                    f"AFQMC observable: {sig_obs:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n"
                )
            elif obs_afqmc is not None:
                print(f"AFQMC observable: {obs_afqmc}\n", flush=True)
            if options["ad_mode"] == "reverse":
                # avg_rdm1 = np.einsum('i,i...->...', global_block_weights, global_block_rdm1s) / np.sum(global_block_weights)
                norms_rdm1 = np.array(list(map(np.linalg.norm, global_block_rdm1s)))
                samples_clean, idx = stat_utils.reject_outliers(
                    np.stack((global_block_weights, norms_rdm1)).T, 1
                )
                global_block_weights = samples_clean[:, 0]
                global_block_rdm1s = global_block_rdm1s[idx]
                avg_rdm1 = np.einsum(
                    "i,i...->...", global_block_weights, global_block_rdm1s
                ) / np.sum(global_block_weights)
                errors_rdm1 = np.array(
                    list(map(np.linalg.norm, global_block_rdm1s - avg_rdm1))
                ) / np.linalg.norm(avg_rdm1)
                print(f"# RDM noise:", flush=True)
                obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                    global_block_weights, errors_rdm1, neql=0, printQ=True
                )
                np.savez("rdm1_afqmc.npz", rdm1=avg_rdm1)
            elif options["ad_mode"] == "2rdm":
                norms_rdm2 = np.array(list(map(np.linalg.norm, global_block_rdm2s)))
                samples_clean, idx = stat_utils.reject_outliers(
                    np.stack((global_block_weights, norms_rdm2)).T, 1
                )
                global_block_weights = samples_clean[:, 0]
                global_block_rdm2s = global_block_rdm2s[idx]
                avg_rdm2 = np.einsum(
                    "i,i...->...", global_block_weights, global_block_rdm2s
                ) / np.sum(global_block_weights)
                errors_rdm2 = np.array(
                    list(map(np.linalg.norm, global_block_rdm2s - avg_rdm2))
                ) / np.linalg.norm(avg_rdm2)
                print(f"# 2RDM noise:", flush=True)
                obs_afqmc, err_afqmc = stat_utils.blocking_analysis(
                    global_block_weights, errors_rdm2, neql=0, printQ=True
                )
                np.savez(
                    "rdm2_afqmc.npz",
                    rdm2=2 * avg_rdm2.reshape(ham.norb, ham.norb, ham.norb, ham.norb),
                )

    comm.Barrier()
    e_afqmc = comm.bcast(e_afqmc, root=0)
    e_err_afqmc = comm.bcast(e_err_afqmc, root=0)
    comm.Barrier()
    return e_afqmc, e_err_afqmc


def fp_afqmc(
    ham_data: dict,
    ham: hamiltonian.hamiltonian,
    propagator: propagation.propagator,
    trial: wavefunctions.wave_function,
    wave_data: dict,
    sampler: sampling.sampler,
    observable,
    options: dict,
    init_walkers=None,
):
    init = time.time()
    seed = options["seed"]

    trial_rdm1 = trial.get_rdm1(wave_data)
    if "rdm1" not in wave_data:
        wave_data["rdm1"] = trial_rdm1
    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, propagator, trial, wave_data
    )
    prop_data = propagator.init_prop_data(trial, wave_data, ham, ham_data, init_walkers)
    prop_data["key"] = random.PRNGKey(seed + rank)

    comm.Barrier()
    init_time = time.time() - init
    if rank == 0:
        print("#\n# Sampling sweeps:")
        print("#  Iter        Mean energy          Stochastic error       Walltime")
    comm.Barrier()

    global_block_weights = np.zeros(size * sampler.n_ene_blocks) + 0.0j
    global_block_energies = np.zeros(size * sampler.n_ene_blocks) + 0.0j

    total_energy = np.zeros(sampler.n_blocks) + 0.0j
    total_weight = np.zeros(sampler.n_blocks) + 0.0j
    for n in range(
        sampler.n_ene_blocks
    ):  # hacking this variable for number of trajectories
        (
            prop_data_tr,
            energy_samples,
            weights,
            prop_data["key"],
        ) = sampler.propagate_free(
            ham, ham_data, propagator, prop_data, trial, wave_data
        )
        global_block_weights[n] = weights[0]
        global_block_energies[n] = energy_samples[0]
        total_weight += weights
        total_energy += weights * (energy_samples - total_energy) / total_weight
        if options["save_walkers"] == True:
            if n > 0:
                with open(f"prop_data_{rank}.bin", "ab") as f:
                    pickle.dump(prop_data_tr, f)
            else:
                with open(f"prop_data_{rank}.bin", "wb") as f:
                    pickle.dump(prop_data_tr, f)

        if n % (max(sampler.n_ene_blocks // 10, 1)) == 0:
            comm.Barrier()
            if rank == 0:
                print(f"{n:5d}: {total_energy}")
        np.savetxt(
            "samples_raw.dat", np.stack((global_block_weights, global_block_energies)).T
        )
