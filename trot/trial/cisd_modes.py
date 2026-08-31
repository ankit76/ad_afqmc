from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System
from .mode_factorization import format_dense_memory_selection


ModeSolver = Literal["auto", "dense", "lanczos"]


@dataclass(frozen=True)
class CisdModeFactorization:
    """Host-side spectral factorization of a restricted spin-adapted K matrix."""

    eigenvalues: np.ndarray
    modes: np.ndarray
    solver: Literal["dense", "lanczos"]
    threshold: float | None
    discarded_norm_target: float | None
    discarded_norm_fraction: float
    natural_rank: int

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def pair_dim(self) -> int:
        return int(np.prod(self.modes.shape[1:]))


def _restricted_k_matrix(amplitudes: np.ndarray) -> tuple[np.ndarray, int, int]:
    amplitudes = np.asarray(amplitudes)
    if amplitudes.ndim != 4:
        raise ValueError(f"restricted doubles amplitudes must have rank 4, got {amplitudes.shape}.")
    nocc, nvir, nocc_2, nvir_2 = amplitudes.shape
    if (nocc_2, nvir_2) != (nocc, nvir):
        raise ValueError(
            "restricted doubles amplitudes must have shape "
            f"{(nocc, nvir, nocc, nvir)}, got {amplitudes.shape}."
        )
    if np.iscomplexobj(amplitudes):
        raise ValueError("restricted mode factorization currently requires real amplitudes.")
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    # Form K_(ia,jb) = 2 A_(ia,jb) - A_(ib,ja) one occupied slab at a
    # time, avoiding a second full exchange-permuted amplitude tensor.
    kernel_4 = np.empty_like(amplitudes)
    for occupied in range(nocc):
        slab = amplitudes[occupied]
        kernel_4[occupied] = 2.0 * slab - slab.transpose(2, 1, 0)
    return kernel_4.reshape(nocc * nvir, nocc * nvir), nocc, nvir


def factorize_cisd_k_modes(
    amplitudes: np.ndarray,
    *,
    threshold: float | None = None,
    discarded_norm_target: float | None = None,
    minimum_rank: int = 0,
    solver: ModeSolver = "auto",
    dense_max_dim: int | None = None,
    lanczos_initial_rank: int = 256,
    lanczos_tol: float = 1.0e-9,
    lanczos_maxiter: int | None = None,
    verbose: bool = False,
) -> CisdModeFactorization:
    """Factor a restricted CISD or raw-T2 spin-adapted kernel on the host.

    With ``solver="auto"``, dense diagonalization is used whenever the
    estimated eigensolver peak fits the currently available host/cgroup
    memory. Otherwise, an adaptive largest-magnitude Lanczos solve is used.
    The automatic dense path also checks LAPACK workspace-index limits and
    retries an inexact selection with Lanczos if the dense solve raises
    :class:`MemoryError`. ``dense_max_dim`` is accepted for compatibility but
    no longer limits automatic selection.
    """

    if threshold is None and discarded_norm_target is None:
        raise ValueError("supply threshold, discarded_norm_target, or both.")
    if threshold is not None and threshold < 0.0:
        raise ValueError("mode threshold must be nonnegative.")
    if discarded_norm_target is not None and not 0.0 <= discarded_norm_target < 1.0:
        raise ValueError("discarded_norm_target must lie in [0, 1).")
    if solver not in ("auto", "dense", "lanczos"):
        raise ValueError("solver must be 'auto', 'dense', or 'lanczos'.")
    if dense_max_dim is not None and dense_max_dim <= 0:
        raise ValueError("dense_max_dim must be positive when provided.")
    if lanczos_initial_rank <= 0:
        raise ValueError("lanczos_initial_rank must be positive.")
    if lanczos_tol <= 0.0:
        raise ValueError("lanczos_tol must be positive.")
    if lanczos_maxiter is not None and lanczos_maxiter <= 0:
        raise ValueError("lanczos_maxiter must be positive when provided.")

    kernel, nocc, nvir = _restricted_k_matrix(amplitudes)
    pair_dim = int(kernel.shape[0])
    if not 0 <= minimum_rank <= pair_dim:
        raise ValueError(f"minimum_rank must lie in [0, {pair_dim}], got {minimum_rank}.")
    full_norm_sq = float(np.vdot(kernel, kernel).real)
    full_norm = float(np.sqrt(full_norm_sq))

    difference_norm_sq = 0.0
    row_batch_size = 256
    for start in range(0, pair_dim, row_batch_size):
        stop = min(start + row_batch_size, pair_dim)
        difference = kernel[start:stop] - kernel[:, start:stop].T
        difference_norm_sq += float(np.vdot(difference, difference).real)
    relative_error = float(np.sqrt(difference_norm_sq / full_norm_sq)) if full_norm_sq else 0.0
    if relative_error > 1.0e-10:
        raise ValueError(
            "restricted spin-adapted K matrix is not symmetric: "
            f"relative error={relative_error:.3e}."
        )

    dense_resource_safe = True
    dense_memory_message = ""
    selected_solver: Literal["dense", "lanczos"]
    if solver == "auto":
        dense_resource_safe, dense_memory_message = format_dense_memory_selection(pair_dim)
        exact_selection = threshold == 0.0 or discarded_norm_target == 0.0
        if (exact_selection or minimum_rank == pair_dim) and not dense_resource_safe:
            raise MemoryError(
                "exact restricted mode selection requires dense diagonalization, but it "
                "is not safe under the automatic resource checks: "
                f"{dense_memory_message}. Request more host memory or use an ILP64 "
                "LAPACK build."
            )
        selected_solver = (
            "dense"
            if exact_selection or minimum_rank == pair_dim or dense_resource_safe
            else "lanczos"
        )
    else:
        selected_solver = solver
    if selected_solver == "lanczos" and (
        threshold == 0.0 or discarded_norm_target == 0.0 or minimum_rank == pair_dim
    ):
        raise ValueError("exact restricted mode selection requires the dense solver.")

    if verbose:
        print(
            "[modes] restricted K factorization: "
            f"pair_dim={pair_dim}, solver={selected_solver}, "
            f"threshold={threshold if threshold is not None else 'none'}, "
            "discarded_norm_target="
            f"{discarded_norm_target if discarded_norm_target is not None else 'none'}, "
            f"minimum_rank={minimum_rank}"
        )
        if solver == "auto":
            print(f"[modes] dense auto-selection resource estimate: {dense_memory_message}")
            if dense_max_dim is not None:
                print("[modes] dense_max_dim is deprecated and ignored by solver='auto'.")

    required_retained_norm_sq = (
        (1.0 - discarded_norm_target**2) * full_norm_sq
        if discarded_norm_target is not None
        else None
    )
    if selected_solver == "dense" or pair_dim <= 2:
        selected_solver = "dense"
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        except MemoryError:
            can_retry_lanczos = (
                solver == "auto"
                and pair_dim > 2
                and threshold != 0.0
                and discarded_norm_target != 0.0
                and minimum_rank != pair_dim
            )
            if not can_retry_lanczos:
                raise
            selected_solver = "lanczos"
            if verbose:
                print(
                    "[modes] dense diagonalization raised MemoryError; "
                    "retrying with Lanczos"
                )
    if selected_solver == "lanczos":
        from scipy.sparse.linalg import LinearOperator, eigsh

        operator = LinearOperator(
            shape=kernel.shape,
            matvec=lambda vector: kernel @ vector,
            matmat=lambda vectors: kernel @ vectors,
            dtype=np.dtype(np.float64),
        )
        requested_rank = min(max(lanczos_initial_rank, minimum_rank), pair_dim - 1)
        rng = np.random.default_rng(91_733)
        v0 = rng.standard_normal(pair_dim)
        while True:
            if verbose:
                print(
                    "[modes] restricted Lanczos largest-magnitude solve: "
                    f"requested_rank={requested_rank}"
                )
            eigenvalues, eigenvectors = eigsh(
                operator,
                k=requested_rank,
                which="LM",
                tol=lanczos_tol,
                maxiter=lanczos_maxiter,
                v0=v0,
            )
            threshold_satisfied = threshold is None or (
                np.count_nonzero(np.abs(eigenvalues) > threshold) < requested_rank
            )
            retained_norm_satisfied = required_retained_norm_sq is None or (
                float(np.vdot(eigenvalues, eigenvalues).real) >= required_retained_norm_sq
            )
            if threshold_satisfied and retained_norm_satisfied:
                break
            if requested_rank == pair_dim - 1:
                raise RuntimeError(
                    "Lanczos could not satisfy restricted mode selection before reaching "
                    "pair_dim - 1 modes; use solver='dense' or relax the selection criteria."
                )
            requested_rank = min(pair_dim - 1, 2 * requested_rank)

    eigenvalues_all = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors_all = np.asarray(eigenvectors, dtype=np.float64)
    order = np.argsort(np.abs(eigenvalues_all))[::-1]
    ordered_eigenvalues = eigenvalues_all[order]
    threshold_rank = (
        int(np.count_nonzero(np.abs(ordered_eigenvalues) > threshold))
        if threshold is not None
        else 0
    )
    norm_rank = 0
    if required_retained_norm_sq is not None and full_norm_sq > 0.0:
        cumulative_norm_sq = np.cumsum(ordered_eigenvalues**2, dtype=np.float64)
        norm_rank = int(
            np.searchsorted(
                cumulative_norm_sq,
                min(required_retained_norm_sq, cumulative_norm_sq[-1]),
                side="left",
            )
            + 1
        )
    natural_rank = max(threshold_rank, norm_rank)
    if natural_rank == 0 and discarded_norm_target is not None:
        natural_rank = 1
    retained_rank = max(natural_rank, minimum_rank)
    if retained_rank == 0:
        raise ValueError("mode selection removed every restricted mode.")
    if retained_rank > ordered_eigenvalues.size:
        raise RuntimeError(
            f"mode selection requires {retained_rank} eigenpairs but only "
            f"{ordered_eigenvalues.size} were computed."
        )

    retained_indices = order[:retained_rank]
    eigenvalues = np.asarray(eigenvalues_all[retained_indices], dtype=np.float64)
    modes = np.asarray(eigenvectors_all[:, retained_indices].T, dtype=np.float64).reshape(
        retained_rank, nocc, nvir
    )
    retained_norm_sq = float(np.vdot(eigenvalues, eigenvalues).real)
    discarded_norm_fraction = (
        float(np.sqrt(max(0.0, full_norm_sq - retained_norm_sq)) / full_norm)
        if full_norm
        else 0.0
    )
    if verbose:
        print(
            "[modes] retained restricted K modes: "
            f"rank={retained_rank}/{pair_dim}, storage={modes.nbytes / 1024**3:.3f} GiB, "
            f"discarded_norm_fraction={discarded_norm_fraction:.3e}, "
            f"natural_rank={natural_rank}"
        )
    return CisdModeFactorization(
        eigenvalues=eigenvalues,
        modes=modes,
        solver=selected_solver,
        threshold=float(threshold) if threshold is not None else None,
        discarded_norm_target=discarded_norm_target,
        discarded_norm_fraction=discarded_norm_fraction,
        natural_rank=natural_rank,
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdModeTrial:
    """Restricted CISD trial stored as a spectral representation of its K matrix.

    The spin-adapted doubles kernel is

        K_(ia,jb) = 2 c_(ia,jb) - c_(ib,ja) = V diag(eigenvalues) V.T.

    ``modes[r, i, a]`` stores row ``r`` of ``V.T``. Retaining every pair-space
    mode gives the exact K matrix; retaining a prefix defines a truncated
    approximation.
    """

    ci1: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array
    nocc_t_core: int = 0
    nvir_t_outer: int = 0

    def __post_init__(self) -> None:
        # Transformations such as ``vmap(..., in_axes=(0, None))`` may rebuild
        # a registered PyTree once with opaque placeholder leaves while
        # inferring axes. Shape validation applies only to actual array leaves.
        if not all(hasattr(value, "ndim") for value in (self.ci1, self.eigenvalues, self.modes)):
            return

        if self.ci1.ndim != 2:
            raise ValueError(f"ci1 must have rank 2, got shape {self.ci1.shape}.")
        if self.eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must have rank 1, got shape {self.eigenvalues.shape}.")
        if self.modes.ndim != 3:
            raise ValueError(f"modes must have rank 3, got shape {self.modes.shape}.")

        pair_dim = self.nocc * self.nvir
        mode_rank = int(self.modes.shape[0])
        if not 0 < mode_rank <= pair_dim:
            raise ValueError(f"mode rank must lie in [1, {pair_dim}], got {mode_rank}.")
        expected_pair_shape = (self.nocc, self.nvir)
        if self.modes.shape[1:] != expected_pair_shape:
            raise ValueError(
                f"each mode must have shape {expected_pair_shape}, got {self.modes.shape[1:]}."
            )
        if self.eigenvalues.shape != (mode_rank,):
            raise ValueError(
                f"eigenvalues must have shape {(mode_rank,)}, got {self.eigenvalues.shape}."
            )
        if self.nocc_t_core < 0 or self.nvir_t_outer < 0:
            raise ValueError("nocc_t_core and nvir_t_outer must be nonnegative.")

    @property
    def nocc(self) -> int:
        """Number of active occupied orbitals."""
        return int(self.ci1.shape[0])

    @property
    def nvir(self) -> int:
        """Number of active virtual orbitals."""
        return int(self.ci1.shape[1])

    @property
    def nocc_full(self) -> int:
        """Number of occupied orbitals in the full AFQMC correlation space."""
        return int(self.nocc_t_core + self.nocc)

    @property
    def nvir_full(self) -> int:
        """Number of virtual orbitals in the full AFQMC correlation space."""
        return int(self.nvir + self.nvir_t_outer)

    @property
    def norb(self) -> int:
        """Number of orbitals in the full AFQMC correlation space."""
        return int(self.nocc_full + self.nvir_full)

    @property
    def occ_act_slice(self) -> slice:
        return slice(self.nocc_t_core, self.nocc_full)

    @property
    def vir_act_slice(self) -> slice:
        return slice(self.nocc_full, self.nocc_full + self.nvir)

    @property
    def norb_act(self) -> int:
        return int(self.nocc + self.nvir)

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        children = (self.ci1, self.eigenvalues, self.modes)
        aux = (self.nocc_t_core, self.nvir_t_outer)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        nocc_t_core, nvir_t_outer = aux
        ci1, eigenvalues, modes = children
        return cls(
            ci1=ci1,
            eigenvalues=eigenvalues,
            modes=modes,
            nocc_t_core=nocc_t_core,
            nvir_t_outer=nvir_t_outer,
        )


def get_rdm1(trial_data: CisdModeTrial) -> jax.Array:
    """Return the restricted reference determinant density matrix."""
    norb, nocc = trial_data.norb, trial_data.nocc_full
    occ = jnp.arange(norb) < nocc
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


def mode_projections(trial_data: CisdModeTrial, matrix: jax.Array) -> jax.Array:
    """Return ``V.T @ matrix``"""
    if matrix.shape != (trial_data.nocc, trial_data.nvir):
        raise ValueError(
            f"matrix must have shape {(trial_data.nocc, trial_data.nvir)}, got {matrix.shape}."
        )

    modes = trial_data.modes
    matrix_r = jnp.real(matrix).astype(modes.dtype)
    projections_r = jnp.einsum("rpt,pt->r", modes, matrix_r, optimize="optimal")
    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        return projections_r.astype(jnp.float64)

    matrix_i = jnp.imag(matrix).astype(modes.dtype)
    projections_i = jnp.einsum("rpt,pt->r", modes, matrix_i, optimize="optimal")
    return projections_r.astype(jnp.complex128) + 1.0j * projections_i.astype(jnp.complex128)


def mode_quadratic(
    trial_data: CisdModeTrial,
    matrix: jax.Array,
    projections: jax.Array | None = None,
) -> jax.Array:
    """Return the bilinear CISD doubles contraction ``matrix.T @ K @ matrix``."""
    if projections is None:
        projections = mode_projections(trial_data, matrix)
    return jnp.sum(
        trial_data.eigenvalues.astype(jnp.float64) * projections * projections,
        dtype=jnp.complex128 if jnp.issubdtype(matrix.dtype, jnp.complexfloating) else jnp.float64,
    )


def mode_apply(
    trial_data: CisdModeTrial,
    matrix: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(V.T @ matrix, K @ matrix)`` with promoted reconstruction."""
    projections = mode_projections(trial_data, matrix)
    values = trial_data.eigenvalues.astype(jnp.float64)
    modes = trial_data.modes

    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        applied = jnp.einsum(
            "r,rqu->qu",
            values * projections.astype(jnp.float64),
            modes,
            optimize="optimal",
        )
        return projections, applied

    applied_r = jnp.einsum(
        "r,rqu->qu",
        values * jnp.real(projections),
        modes,
        optimize="optimal",
    )
    applied_i = jnp.einsum(
        "r,rqu->qu",
        values * jnp.imag(projections),
        modes,
        optimize="optimal",
    )
    return projections, applied_r.astype(jnp.complex128) + 1.0j * applied_i.astype(jnp.complex128)


def overlap_r(walker: jax.Array, trial_data: CisdModeTrial) -> jax.Array:
    """Overlap for a restricted walker and CISD trial."""
    nocc_full = trial_data.nocc_full
    wocc = walker[:nocc_full, :]
    green = jnp.linalg.solve(wocc.T, walker.T)
    det0 = jnp.linalg.det(wocc)

    green_occ = green[trial_data.occ_act_slice, trial_data.vir_act_slice]
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    doubles = mode_quadratic(trial_data, green_occ)
    return (1.0 + 2.0 * ci1g + doubles) * det0 * det0


def make_cisd_mode_trial_data(
    data: dict,
    sys: System,
    *,
    mixed_precision: bool = True,
    mode_threshold: float | None = 0.0,
    discarded_norm_target: float | None = None,
    minimum_rank: int = 0,
    mode_solver: ModeSolver = "auto",
    dense_max_dim: int | None = None,
    lanczos_initial_rank: int = 256,
    lanczos_tol: float = 1.0e-9,
    lanczos_maxiter: int | None = None,
    verbose: bool = False,
) -> CisdModeTrial:
    """Build a full-rank or truncated mode-native restricted CISD trial.

    ``modes`` has shape ``(rank, nocc, nvir)``. For compatibility with
    eigensolver output, ``eigenvectors`` with shape ``(pair_dim, rank)`` is
    also accepted and interpreted as eigenvectors stored in columns. If modes
    are not supplied, ``ci2`` is factorized on the host using the same
    memory-aware dense/Lanczos policy as PT-RCC modes.
    """
    del sys

    ci1 = jnp.asarray(data["ci1"], dtype=jnp.float64)
    nocc, nvir = ci1.shape
    pair_dim = int(nocc * nvir)
    if "eigenvalues" in data:
        eigenvalues = jnp.asarray(data["eigenvalues"], dtype=jnp.float64)
        if eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must have rank 1, got shape {eigenvalues.shape}.")
        mode_rank = int(eigenvalues.shape[0])
        if "modes" in data:
            modes_input = jnp.asarray(data["modes"])
        elif "eigenvectors" in data:
            eigenvectors = jnp.asarray(data["eigenvectors"])
            expected_eigenvector_shape = (pair_dim, mode_rank)
            if eigenvectors.shape != expected_eigenvector_shape:
                raise ValueError(
                    f"eigenvectors must have shape {expected_eigenvector_shape}, "
                    f"got {eigenvectors.shape}."
                )
            modes_input = eigenvectors.T.reshape(mode_rank, nocc, nvir)
        else:
            raise KeyError("mode-native CISD trial data requires 'modes' or 'eigenvectors'.")
    else:
        if "ci2" not in data:
            raise KeyError("mode-native CISD trial data requires modes or dense 'ci2'.")
        factorization = factorize_cisd_k_modes(
            np.asarray(data["ci2"]),
            threshold=mode_threshold,
            discarded_norm_target=discarded_norm_target,
            minimum_rank=minimum_rank,
            solver=mode_solver,
            dense_max_dim=dense_max_dim,
            lanczos_initial_rank=lanczos_initial_rank,
            lanczos_tol=lanczos_tol,
            lanczos_maxiter=lanczos_maxiter,
            verbose=verbose,
        )
        eigenvalues = jnp.asarray(factorization.eigenvalues, dtype=jnp.float64)
        modes_input = jnp.asarray(factorization.modes)

    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    modes = jnp.asarray(modes_input, dtype=mode_dtype)
    nocc_t_core = int(jnp.asarray(data.get("nocc_t_core", 0)).item())
    nvir_t_outer = int(jnp.asarray(data.get("nvir_t_outer", 0)).item())
    return CisdModeTrial(
        ci1=ci1,
        eigenvalues=eigenvalues,
        modes=modes,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )


def make_cisd_mode_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("Restricted CISD mode trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD mode trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
