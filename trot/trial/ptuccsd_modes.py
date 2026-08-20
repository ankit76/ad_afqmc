from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System
from .ptuccsd_thouless import thouless_mo_from_t1
from .ucisd_k_modes import factorize_ucisd_k_blocks


@dataclass(frozen=True)
class PtuccsdModeFactorization:
    """Host-side spectral factorization of the combined raw-``T2`` kernel."""

    eigenvalues: np.ndarray
    modes: np.ndarray
    pair_dims: tuple[int, int]
    solver: Literal["dense", "lanczos"]
    threshold: float
    discarded_norm_fraction: float

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def combined_dim(self) -> int:
        return int(sum(self.pair_dims))


def factorize_t2_modes(
    t2aa: np.ndarray,
    t2ab: np.ndarray,
    t2bb: np.ndarray,
    *,
    mode_threshold: float = 0.0,
    solver: Literal["auto", "dense", "lanczos"] = "auto",
    dense_max_dim: int = 2048,
    lanczos_initial_rank: int = 256,
    lanczos_tol: float = 1.0e-9,
    lanczos_maxiter: int | None = None,
    verbose: bool = False,
) -> PtuccsdModeFactorization:
    """Factorize the combined spin-pair kernel formed from raw UCCSD ``T2``.

    The kernel acts on ``z = concatenate((g_alpha, g_beta))`` and is

    ``K = [[T2aa, T2ab], [T2ab.T, T2bb]]``.

    The same-spin blocks are flattened over occupied--virtual pairs.  A zero
    threshold gives an exact full-rank representation and therefore requires
    the dense solver.  For a positive threshold, ``solver="auto"`` can use the
    matrix-free Lanczos implementation already validated for UCISD kernels.
    """

    for name, block in (("t2aa", t2aa), ("t2ab", t2ab), ("t2bb", t2bb)):
        if np.iscomplexobj(block):
            raise ValueError(f"{name} must be real for PT-UCCSD mode factorization.")

    factorization = factorize_ucisd_k_blocks(
        t2aa,
        t2ab,
        t2bb,
        threshold=mode_threshold,
        solver=solver,
        dense_max_dim=dense_max_dim,
        lanczos_initial_rank=lanczos_initial_rank,
        lanczos_tol=lanczos_tol,
        lanczos_maxiter=lanczos_maxiter,
        verbose=verbose,
    )
    return PtuccsdModeFactorization(
        eigenvalues=factorization.eigenvalues,
        modes=factorization.modes,
        pair_dims=factorization.pair_dims,
        solver=factorization.solver,
        threshold=factorization.threshold,
        discarded_norm_fraction=factorization.discarded_norm_fraction,
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdThoulessModeTrial:
    """PT-UCCSD exponential trial stored as combined raw-``T2`` modes.

    For occupied--virtual Green-function blocks ``g_alpha`` and ``g_beta``,
    let ``z = concatenate((g_alpha, g_beta))``.  The retained modes represent

    ``K ~= modes.T @ diag(eigenvalues) @ modes``

    and the exponential doubles coordinate is ``theta2 = 0.5 * z.T @ K @ z``.
    The modes may mix alpha and beta pair spaces.  The spin-dependent Thouless
    references and orbital bases remain explicit, while restricted walkers
    continue to share their first ``min(n_alpha, n_beta)`` occupied columns.
    """

    mo_t_a: jax.Array
    mo_t_b: jax.Array
    mo_coeff_b: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array

    def __post_init__(self) -> None:
        arrays = (
            self.mo_t_a,
            self.mo_t_b,
            self.mo_coeff_b,
            self.eigenvalues,
            self.modes,
        )
        if not all(hasattr(value, "ndim") for value in arrays):
            return
        if self.mo_t_a.ndim != 2 or self.mo_t_b.ndim != 2:
            raise ValueError("mo_t_a and mo_t_b must have rank 2.")
        if self.mo_t_a.shape[0] != self.mo_t_b.shape[0]:
            raise ValueError(
                "mo_t_a and mo_t_b must use the same orbital dimension, got "
                f"{self.mo_t_a.shape} and {self.mo_t_b.shape}."
            )
        if self.mo_coeff_b.shape != (self.norb, self.norb):
            raise ValueError(
                f"mo_coeff_b must have shape {(self.norb, self.norb)}, "
                f"got {self.mo_coeff_b.shape}."
            )
        if self.eigenvalues.ndim != 1:
            raise ValueError(
                f"eigenvalues must have rank 1, got shape {self.eigenvalues.shape}."
            )
        if self.modes.ndim != 2:
            raise ValueError(f"modes must have rank 2, got shape {self.modes.shape}.")
        expected = (self.mode_rank, sum(self.pair_dim))
        if self.modes.shape != expected:
            raise ValueError(f"modes must have shape {expected}, got {self.modes.shape}.")
        if jnp.issubdtype(self.modes.dtype, jnp.complexfloating):
            raise ValueError("PtuccsdThoulessModeTrial currently requires real modes.")

    @property
    def norb(self) -> int:
        return int(self.mo_t_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.mo_t_a.shape[1]), int(self.mo_t_b.shape[1]))

    @property
    def nvir(self) -> tuple[int, int]:
        noa, nob = self.nocc
        return (self.norb - noa, self.norb - nob)

    @property
    def pair_dim(self) -> tuple[int, int]:
        return (
            int(self.nocc[0] * self.nvir[0]),
            int(self.nocc[1] * self.nvir[1]),
        )

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        return (
            self.mo_t_a,
            self.mo_t_b,
            self.mo_coeff_b,
            self.eigenvalues,
            self.modes,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def mode_projections(
    trial_data: PtuccsdThoulessModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
) -> jax.Array:
    """Project alpha and beta pair matrices onto the combined real modes."""

    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )

    da, _ = trial_data.pair_dim
    modes_a = trial_data.modes[:, :da]
    modes_b = trial_data.modes[:, da:]
    vector_a_r = jnp.real(matrix_a).reshape(-1).astype(trial_data.modes.dtype)
    vector_b_r = jnp.real(matrix_b).reshape(-1).astype(trial_data.modes.dtype)
    projection_a_r = jnp.einsum("rp,p->r", modes_a, vector_a_r, optimize="optimal")
    projection_b_r = jnp.einsum("rp,p->r", modes_b, vector_b_r, optimize="optimal")
    projection_r = projection_a_r.astype(jnp.float64) + projection_b_r.astype(jnp.float64)

    is_complex = jnp.issubdtype(matrix_a.dtype, jnp.complexfloating) or jnp.issubdtype(
        matrix_b.dtype, jnp.complexfloating
    )
    if not is_complex:
        return projection_r

    vector_a_i = jnp.imag(matrix_a).reshape(-1).astype(trial_data.modes.dtype)
    vector_b_i = jnp.imag(matrix_b).reshape(-1).astype(trial_data.modes.dtype)
    projection_a_i = jnp.einsum("rp,p->r", modes_a, vector_a_i, optimize="optimal")
    projection_b_i = jnp.einsum("rp,p->r", modes_b, vector_b_i, optimize="optimal")
    projection_i = projection_a_i.astype(jnp.float64) + projection_b_i.astype(jnp.float64)
    return projection_r.astype(jnp.complex128) + 1.0j * projection_i.astype(jnp.complex128)


def mode_apply(
    trial_data: PtuccsdThoulessModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    projections: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply the retained combined kernel and split the result by spin."""

    if projections is None:
        projections = mode_projections(trial_data, matrix_a, matrix_b)
    weighted = trial_data.eigenvalues.astype(jnp.float64) * projections
    if not jnp.issubdtype(projections.dtype, jnp.complexfloating):
        applied = jnp.einsum("r,rp->p", weighted, trial_data.modes, optimize="optimal")
    else:
        applied_r = jnp.einsum(
            "r,rp->p", jnp.real(weighted), trial_data.modes, optimize="optimal"
        )
        applied_i = jnp.einsum(
            "r,rp->p", jnp.imag(weighted), trial_data.modes, optimize="optimal"
        )
        applied = applied_r.astype(jnp.complex128)
        applied += 1.0j * applied_i.astype(jnp.complex128)

    da, _ = trial_data.pair_dim
    shape_a = (trial_data.nocc[0], trial_data.nvir[0])
    shape_b = (trial_data.nocc[1], trial_data.nvir[1])
    return applied[:da].reshape(shape_a), applied[da:].reshape(shape_b)


def mode_quadratic(
    trial_data: PtuccsdThoulessModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    projections: jax.Array | None = None,
) -> jax.Array:
    """Return ``0.5 * z.T @ K_modes @ z`` without complex conjugation."""

    if projections is None:
        projections = mode_projections(trial_data, matrix_a, matrix_b)
    dtype = (
        jnp.complex128
        if jnp.issubdtype(projections.dtype, jnp.complexfloating)
        else jnp.float64
    )
    return 0.5 * jnp.sum(
        trial_data.eigenvalues.astype(jnp.float64) * projections * projections,
        dtype=dtype,
    )


def _projector(coeff: jax.Array) -> jax.Array:
    metric = coeff.conj().T @ coeff
    return coeff @ jnp.linalg.solve(metric, coeff.conj().T)


def get_rdm1(trial_data: PtuccsdThoulessModeTrial) -> jax.Array:
    dm_a = _projector(trial_data.mo_t_a)
    beta_occ_alpha_basis = trial_data.mo_coeff_b @ trial_data.mo_t_b
    dm_b = _projector(beta_occ_alpha_basis)
    return jnp.stack([dm_a, dm_b], axis=0)


def _spin_green(walker: jax.Array, mo_t: jax.Array) -> jax.Array:
    overlap = mo_t.conj().T @ walker
    half_green = jnp.linalg.solve(overlap.T, walker.T)
    return mo_t.conj() @ half_green


def _spin_overlap_and_green_occ(
    walker: jax.Array,
    mo_t: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the determinant overlap and occupied--virtual Green block.

    The exponential guide only needs the occupied--virtual rows of
    ``G = mo_t.conj() @ half_green``. Forming those rows directly avoids the
    full ``(norb, norb)`` Green matrix during every overlap evaluation.
    """

    overlap_matrix = mo_t.conj().T @ walker
    half_green = jnp.linalg.solve(overlap_matrix.T, walker.T)
    nocc = int(mo_t.shape[1])
    green_occ = mo_t.conj()[:nocc, :] @ half_green[:, nocc:]
    return jnp.linalg.det(overlap_matrix), green_occ


def greens_unrestricted(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array]:
    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    return (
        _spin_green(walker_a, trial_data.mo_t_a),
        _spin_green(walker_b_beta, trial_data.mo_t_b),
    )


def theta_t2_from_greens(
    green_a: jax.Array,
    green_b: jax.Array,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return mode_quadratic(
        trial_data,
        green_a[:noa, noa:],
        green_b[:nob, nob:],
    )


def theta_t2_u(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    _, green_occ_a = _spin_overlap_and_green_occ(walker_a, trial_data.mo_t_a)
    _, green_occ_b = _spin_overlap_and_green_occ(walker_b_beta, trial_data.mo_t_b)
    return mode_quadratic(trial_data, green_occ_a, green_occ_b)


def reference_overlap_u(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    overlap_a = trial_data.mo_t_a.conj().T @ walker_a
    overlap_b = trial_data.mo_t_b.conj().T @ walker_b_beta
    return jnp.linalg.det(overlap_a) * jnp.linalg.det(overlap_b)


def overlap_u(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    overlap_a, green_occ_a = _spin_overlap_and_green_occ(
        walker_a,
        trial_data.mo_t_a,
    )
    overlap_b, green_occ_b = _spin_overlap_and_green_occ(
        walker_b_beta,
        trial_data.mo_t_b,
    )
    theta = mode_quadratic(trial_data, green_occ_a, green_occ_b)
    return overlap_a * overlap_b * jnp.exp(theta)


def _split_restricted_walker(
    walker: jax.Array,
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array]:
    noa, nob = trial_data.nocc
    if walker.shape[1] < max(noa, nob):
        raise ValueError(
            "restricted walker has too few occupied columns for the PT-UCCSD mode trial: "
            f"walker shape={walker.shape}, nocc={trial_data.nocc}."
        )
    return walker[:, :noa], walker[:, :nob]


def reference_overlap_r(
    walker: jax.Array,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    return reference_overlap_u(_split_restricted_walker(walker, trial_data), trial_data)


def overlap_r(walker: jax.Array, trial_data: PtuccsdThoulessModeTrial) -> jax.Array:
    return overlap_u(_split_restricted_walker(walker, trial_data), trial_data)


def _t2_to_iajb(t2: np.ndarray, layout: str) -> np.ndarray:
    if layout in {"pyscf", "ijab"}:
        return np.asarray(t2).transpose(0, 2, 1, 3)
    if layout == "iajb":
        return np.asarray(t2)
    raise ValueError(f"Unknown PT-UCCSD t2_layout: {layout!r}")


def make_ptuccsd_thouless_mode_trial_data(
    data: dict,
    sys: System | None = None,
    *,
    mixed_precision: bool = True,
    mode_threshold: float = 0.0,
    mode_solver: Literal["auto", "dense", "lanczos"] = "auto",
    dense_max_dim: int = 2048,
    lanczos_initial_rank: int = 256,
    lanczos_tol: float = 1.0e-9,
    lanczos_maxiter: int | None = None,
    verbose: bool = False,
) -> PtuccsdThoulessModeTrial:
    """Build an exact or truncated combined-mode PT-UCCSD trial.

    Precomputed ``eigenvalues`` and either row-major ``modes`` or column-major
    ``eigenvectors`` may be supplied.  Otherwise the three dense raw-``T2``
    blocks are factorized on the host and are not retained by the trial.
    """

    mo_t_a = (
        jnp.asarray(data["mo_t_a"])
        if "mo_t_a" in data
        else thouless_mo_from_t1(jnp.asarray(data["t1a"]))
    )
    mo_t_b = (
        jnp.asarray(data["mo_t_b"])
        if "mo_t_b" in data
        else thouless_mo_from_t1(jnp.asarray(data["t1b"]))
    )
    mo_coeff_b_raw = data.get("mo_coeff_b", data.get("mo_b"))
    if mo_coeff_b_raw is None:
        raise KeyError("PT-UCCSD mode trial data requires 'mo_coeff_b' or 'mo_b'.")

    noa = int(mo_t_a.shape[1])
    nob = int(mo_t_b.shape[1])
    norb = int(mo_t_a.shape[0])
    combined_dim = noa * (norb - noa) + nob * (norb - nob)
    if "eigenvalues" in data:
        eigenvalues_raw = np.asarray(data["eigenvalues"])
        if np.iscomplexobj(eigenvalues_raw):
            raise ValueError("PT-UCCSD mode eigenvalues must be real.")
        eigenvalues = np.asarray(eigenvalues_raw, dtype=np.float64)
        if eigenvalues.ndim != 1:
            raise ValueError(
                f"eigenvalues must have rank 1, got shape {eigenvalues.shape}."
            )
        rank = int(eigenvalues.shape[0])
        if "modes" in data:
            modes = np.asarray(data["modes"])
        elif "eigenvectors" in data:
            eigenvectors = np.asarray(data["eigenvectors"])
            if eigenvectors.shape != (combined_dim, rank):
                raise ValueError(
                    f"eigenvectors must have shape {(combined_dim, rank)}, "
                    f"got {eigenvectors.shape}."
                )
            modes = eigenvectors.T
        else:
            raise KeyError("PT-UCCSD mode data requires 'modes' or 'eigenvectors'.")
        if np.iscomplexobj(modes):
            raise ValueError("PT-UCCSD modes must be real.")
    else:
        layout = str(data.get("t2_layout", "pyscf")).lower()
        factorization = factorize_t2_modes(
            _t2_to_iajb(data["t2aa"], layout),
            _t2_to_iajb(data["t2ab"], layout),
            _t2_to_iajb(data["t2bb"], layout),
            mode_threshold=mode_threshold,
            solver=mode_solver,
            dense_max_dim=dense_max_dim,
            lanczos_initial_rank=lanczos_initial_rank,
            lanczos_tol=lanczos_tol,
            lanczos_maxiter=lanczos_maxiter,
            verbose=verbose,
        )
        eigenvalues = factorization.eigenvalues
        modes = factorization.modes

    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    trial_data = PtuccsdThoulessModeTrial(
        mo_t_a=mo_t_a,
        mo_t_b=mo_t_b,
        mo_coeff_b=jnp.asarray(mo_coeff_b_raw),
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
    )
    if sys is not None:
        if trial_data.norb != sys.norb:
            raise ValueError(
                "PT-UCCSD mode trial/system orbital mismatch: "
                f"trial norb={trial_data.norb}, system norb={sys.norb}."
            )
        if trial_data.nocc != sys.nelec:
            raise ValueError(
                "PT-UCCSD mode trial/system electron mismatch: "
                f"trial nocc={trial_data.nocc}, system nelec={sys.nelec}."
            )
    return trial_data


def make_ptuccsd_thouless_mode_trial_ops(sys: System) -> TrialOps:
    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT-UCCSD mode trials require nup >= ndn.")
        overlap_fn = overlap_r
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
    else:
        raise ValueError(
            "PT-UCCSD mode trial supports restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_fn, get_rdm1=get_rdm1)


__all__ = [
    "PtuccsdModeFactorization",
    "PtuccsdThoulessModeTrial",
    "factorize_t2_modes",
    "get_rdm1",
    "greens_unrestricted",
    "make_ptuccsd_thouless_mode_trial_data",
    "make_ptuccsd_thouless_mode_trial_ops",
    "mode_apply",
    "mode_projections",
    "mode_quadratic",
    "overlap_r",
    "overlap_u",
    "reference_overlap_r",
    "reference_overlap_u",
    "theta_t2_from_greens",
    "theta_t2_u",
]
