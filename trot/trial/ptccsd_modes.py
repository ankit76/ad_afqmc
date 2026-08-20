from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System
from .cisd_modes import mode_quadratic


def decompose_t2_modes(
    t2: np.ndarray,
    *,
    mode_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the restricted spin-adapted kernel formed from raw ``T2``.

    The returned modes represent

    ``K_(ia,jb) = 2 t_(ia,jb) - t_(ib,ja)``.

    Unlike the CISD-mode construction, ``t2`` must not contain the disconnected
    ``T1*T1`` contribution.
    """

    t2_array = np.asarray(t2)
    if t2_array.ndim != 4:
        raise ValueError(f"t2 must have rank 4, got shape {t2_array.shape}.")
    nocc, nvir, nocc_2, nvir_2 = t2_array.shape
    if (nocc_2, nvir_2) != (nocc, nvir):
        raise ValueError(
            "t2 must have shape (nocc, nvir, nocc, nvir); "
            f"got {t2_array.shape}."
        )
    if np.iscomplexobj(t2_array):
        raise ValueError("Restricted PT mode decomposition currently requires real T2 amplitudes.")
    if mode_threshold < 0.0:
        raise ValueError("mode_threshold must be nonnegative.")

    pair_dim = nocc * nvir
    direct = np.asarray(t2_array, dtype=np.float64).reshape(pair_dim, pair_dim)
    exchange = np.transpose(t2_array, (0, 3, 2, 1)).reshape(pair_dim, pair_dim)
    kernel = 2.0 * direct - exchange
    kernel_norm = float(np.linalg.norm(kernel))
    symmetry_error = (
        float(np.linalg.norm(kernel - kernel.T)) / kernel_norm if kernel_norm > 0.0 else 0.0
    )
    if symmetry_error > 1.0e-10:
        raise ValueError(
            "spin-adapted raw-T2 kernel is not symmetric: "
            f"relative error={symmetry_error:.3e}."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    modes = eigenvectors[:, order].T.reshape(pair_dim, nocc, nvir)
    keep = np.abs(eigenvalues) > mode_threshold
    if not np.any(keep):
        raise ValueError("mode_threshold removed every raw-T2 mode.")
    return eigenvalues[keep], modes[keep]


def _validate_modes(
    eigenvalues: jax.Array,
    modes: jax.Array,
    *,
    nocc: int,
    nvir: int,
) -> None:
    if not all(hasattr(value, "ndim") for value in (eigenvalues, modes)):
        return
    if eigenvalues.ndim != 1:
        raise ValueError(f"eigenvalues must have rank 1, got shape {eigenvalues.shape}.")
    if modes.ndim != 3:
        raise ValueError(f"modes must have rank 3, got shape {modes.shape}.")
    rank = int(eigenvalues.shape[0])
    pair_dim = nocc * nvir
    if not 0 < rank <= pair_dim:
        raise ValueError(f"mode rank must lie in [1, {pair_dim}], got {rank}.")
    if modes.shape != (rank, nocc, nvir):
        raise ValueError(
            f"modes must have shape {(rank, nocc, nvir)}, got {modes.shape}."
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdModeTrial:
    """First-order restricted PT-CCSD importance guide using raw-T2 modes."""

    t1: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.t1, "ndim"):
            return
        if self.t1.ndim != 2:
            raise ValueError(f"t1 must have rank 2, got shape {self.t1.shape}.")
        _validate_modes(
            self.eigenvalues,
            self.modes,
            nocc=self.nocc,
            nvir=self.nvir,
        )

    @property
    def nocc(self) -> int:
        return int(self.t1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t1.shape[1])

    @property
    def norb(self) -> int:
        return self.nocc + self.nvir

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        return (self.t1, self.eigenvalues, self.modes), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        t1, eigenvalues, modes = children
        return cls(t1=t1, eigenvalues=eigenvalues, modes=modes)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessModeTrial:
    """Restricted Thouless reference plus a spectral representation of raw ``T2``."""

    mo_t: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.mo_t, "ndim") or not hasattr(self.modes, "ndim"):
            return
        if self.mo_t.ndim != 2:
            raise ValueError(f"mo_t must have rank 2, got shape {self.mo_t.shape}.")
        nocc = int(self.mo_t.shape[1])
        nvir = int(self.modes.shape[2])
        if self.mo_t.shape[0] != nocc + nvir:
            raise ValueError(
                f"mo_t must have shape {(nocc + nvir, nocc)}, got {self.mo_t.shape}."
            )
        _validate_modes(self.eigenvalues, self.modes, nocc=nocc, nvir=nvir)

    @property
    def nocc(self) -> int:
        return int(self.mo_t.shape[1])

    @property
    def nvir(self) -> int:
        return int(self.modes.shape[2])

    @property
    def norb(self) -> int:
        return self.nocc + self.nvir

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        return (self.mo_t, self.eigenvalues, self.modes), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mo_t, eigenvalues, modes = children
        return cls(mo_t=mo_t, eigenvalues=eigenvalues, modes=modes)


def make_ptccsd_mode_trial_data(
    data: dict,
    *,
    mixed_precision: bool = True,
    mode_threshold: float = 0.0,
) -> PtccsdModeTrial:
    t1 = jnp.asarray(data["t1"], dtype=jnp.float64)
    if "eigenvalues" in data and "modes" in data:
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float64)
        modes = np.asarray(data["modes"])
    else:
        eigenvalues, modes = decompose_t2_modes(
            np.asarray(data["t2"]),
            mode_threshold=mode_threshold,
        )
    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    return PtccsdModeTrial(
        t1=t1,
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
    )


def make_ptccsd_thouless_mode_trial_data(
    data: dict,
    *,
    mixed_precision: bool = True,
    mode_threshold: float = 0.0,
) -> PtccsdThoulessModeTrial:
    if "mo_t" in data:
        mo_t = jnp.asarray(data["mo_t"])
    else:
        t1 = jnp.asarray(data["t1"])
        mo_t = jnp.vstack([jnp.eye(t1.shape[0], dtype=t1.dtype), t1.T])
    if "eigenvalues" in data and "modes" in data:
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float64)
        modes = np.asarray(data["modes"])
    else:
        eigenvalues, modes = decompose_t2_modes(
            np.asarray(data["t2"]),
            mode_threshold=mode_threshold,
        )
    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    return PtccsdThoulessModeTrial(
        mo_t=mo_t,
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
    )


def get_rdm1_pt(trial_data: PtccsdModeTrial) -> jax.Array:
    occ = jnp.arange(trial_data.norb) < trial_data.nocc
    dm = jnp.diag(occ).astype(float)
    return jnp.stack([dm, dm], axis=0)


def get_rdm1_thouless(trial_data: PtccsdThoulessModeTrial) -> jax.Array:
    c = trial_data.mo_t
    dm = c @ jnp.linalg.solve(c.conj().T @ c, c.conj().T)
    return jnp.stack([dm, dm], axis=0)


def greens_pt_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    wocc = walker[: trial_data.nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def hf_overlap_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    det0 = jnp.linalg.det(walker[: trial_data.nocc, :])
    return det0 * det0


def theta_pt_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    green_occ = greens_pt_r(walker, trial_data)[:, trial_data.nocc :]
    theta1 = 2.0 * jnp.einsum("ia,ia->", trial_data.t1, green_occ, optimize="optimal")
    return theta1 + mode_quadratic(trial_data, green_occ)


def overlap_pt_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    return hf_overlap_r(walker, trial_data) * jnp.exp(theta_pt_r(walker, trial_data))


def half_green_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    overlap_mat = trial_data.mo_t.conj().T @ walker
    return jnp.linalg.solve(overlap_mat.T, walker.T)


def _det_overlap_and_green_occ_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array]:
    """Return determinant overlap and the needed Green excitation block."""

    overlap_mat = trial_data.mo_t.conj().T @ walker
    half_green = jnp.linalg.solve(overlap_mat.T, walker.T)
    nocc = trial_data.nocc
    green_occ = trial_data.mo_t.conj()[:nocc, :] @ half_green[:, nocc:]
    return jnp.linalg.det(overlap_mat) ** 2, green_occ


def greens_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    return trial_data.mo_t.conj() @ half_green_thouless_r(walker, trial_data)


def greenp_thouless(
    green: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    return (green - jnp.eye(trial_data.norb, dtype=green.dtype))[:, trial_data.nocc :]


def det_overlap_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    return jnp.linalg.det(trial_data.mo_t.conj().T @ walker) ** 2


def theta_t2_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    _, green_occ = _det_overlap_and_green_occ_thouless_r(walker, trial_data)
    return mode_quadratic(trial_data, green_occ)


def overlap_ptccsd_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    det_overlap, green_occ = _det_overlap_and_green_occ_thouless_r(walker, trial_data)
    return det_overlap * jnp.exp(mode_quadratic(trial_data, green_occ))


def make_ptccsd_mode_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode guide requires a closed-shell restricted walker.")
    return TrialOps(overlap=overlap_pt_r, get_rdm1=get_rdm1_pt)


def make_ptccsd_thouless_mode_trial_ops(
    sys: System,
    *,
    exponentiate_t2: bool = True,
) -> TrialOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD Thouless mode guide requires a closed-shell restricted walker.")
    overlap = overlap_ptccsd_thouless_r if exponentiate_t2 else det_overlap_thouless_r
    return TrialOps(overlap=overlap, get_rdm1=get_rdm1_thouless)
