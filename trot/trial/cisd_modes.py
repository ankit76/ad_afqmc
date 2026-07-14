from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


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
) -> CisdModeTrial:
    """Build a full-rank or truncated mode-native restricted CISD trial.

    ``modes`` has shape ``(rank, nocc, nvir)``. For compatibility with
    eigensolver output, ``eigenvectors`` with shape ``(pair_dim, rank)`` is
    also accepted and interpreted as eigenvectors stored in columns.
    """
    del sys

    ci1 = jnp.asarray(data["ci1"], dtype=jnp.float64)
    eigenvalues = jnp.asarray(data["eigenvalues"], dtype=jnp.float64)
    if eigenvalues.ndim != 1:
        raise ValueError(f"eigenvalues must have rank 1, got shape {eigenvalues.shape}.")
    nocc, nvir = ci1.shape
    pair_dim = int(nocc * nvir)
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
