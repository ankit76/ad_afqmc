from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


def build_spin_adapted_kernel(ci2: jax.Array) -> jax.Array:
    """Return the restricted CISD pair-space kernel ``K = 2 C - C_ex``.

    ``ci2`` is ordered as ``ci2[i, a, j, b]`` and the returned matrix is
    ordered by the flattened pair indices ``(i, a)`` and ``(j, b)``.
    """
    ci2 = jnp.asarray(ci2)
    if ci2.ndim != 4:
        raise ValueError(f"ci2 must have rank 4, got shape {ci2.shape}.")
    nocc, nvir, nocc_2, nvir_2 = ci2.shape
    if (nocc_2, nvir_2) != (nocc, nvir):
        raise ValueError(f"ci2 must have shape (nocc, nvir, nocc, nvir), got {ci2.shape}.")

    pair_dim = int(nocc * nvir)
    direct = ci2.reshape(pair_dim, pair_dim)
    exchange = jnp.transpose(ci2, (0, 3, 2, 1)).reshape(pair_dim, pair_dim)
    return 2.0 * direct - exchange


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdKTrial:
    """Restricted CISD trial stored directly as its spin-adapted K matrix.

    The dense doubles tensor is replaced by

    ``K_(ia,jb) = 2 c_(ia,jb) - c_(ib,ja)``.

    This is an opt-in representation intended for exact full-rank overlap,
    force-bias, and local-energy evaluations without retaining ``ci2``.
    """

    ci1: jax.Array
    k: jax.Array
    nocc_t_core: int = 0
    nvir_t_outer: int = 0

    def __post_init__(self) -> None:
        # JAX transformations may reconstruct registered PyTrees once with
        # opaque placeholder leaves while inferring axes.
        if not all(hasattr(value, "ndim") for value in (self.ci1, self.k)):
            return

        if self.ci1.ndim != 2:
            raise ValueError(f"ci1 must have rank 2, got shape {self.ci1.shape}.")
        if self.k.ndim != 2:
            raise ValueError(f"k must have rank 2, got shape {self.k.shape}.")
        pair_dim = self.nocc * self.nvir
        if self.k.shape != (pair_dim, pair_dim):
            raise ValueError(f"k must have shape {(pair_dim, pair_dim)}, got {self.k.shape}.")
        if jnp.issubdtype(self.k.dtype, jnp.complexfloating):
            raise ValueError("CisdKTrial currently requires a real K matrix.")
        if self.nocc_t_core < 0 or self.nvir_t_outer < 0:
            raise ValueError("nocc_t_core and nvir_t_outer must be nonnegative.")

    @property
    def nocc(self) -> int:
        return int(self.ci1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.ci1.shape[1])

    @property
    def nocc_full(self) -> int:
        return int(self.nocc_t_core + self.nocc)

    @property
    def nvir_full(self) -> int:
        return int(self.nvir + self.nvir_t_outer)

    @property
    def norb(self) -> int:
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
    def pair_dim(self) -> int:
        return int(self.nocc * self.nvir)

    def tree_flatten(self):
        children = (self.ci1, self.k)
        aux = (self.nocc_t_core, self.nvir_t_outer)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        nocc_t_core, nvir_t_outer = aux
        ci1, k = children
        return cls(
            ci1=ci1,
            k=k,
            nocc_t_core=nocc_t_core,
            nvir_t_outer=nvir_t_outer,
        )


def get_rdm1(trial_data: CisdKTrial) -> jax.Array:
    """Return the restricted reference determinant density matrix."""
    occ = jnp.arange(trial_data.norb) < trial_data.nocc_full
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


def k_apply(trial_data: CisdKTrial, matrix: jax.Array) -> jax.Array:
    """Return ``K @ matrix`` using the stored K precision.

    K is real, so real and imaginary components are contracted separately.
    The operation is bilinear; no complex conjugation is applied.
    """
    expected_shape = (trial_data.nocc, trial_data.nvir)
    if matrix.shape != expected_shape:
        raise ValueError(f"matrix must have shape {expected_shape}, got {matrix.shape}.")

    matrix_r = jnp.real(matrix).reshape(-1).astype(trial_data.k.dtype)
    applied_r = jnp.einsum("pq,q->p", trial_data.k, matrix_r, optimize="optimal")
    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        return applied_r.reshape(expected_shape)

    matrix_i = jnp.imag(matrix).reshape(-1).astype(trial_data.k.dtype)
    applied_i = jnp.einsum("pq,q->p", trial_data.k, matrix_i, optimize="optimal")
    complex_dtype = jnp.complex64 if trial_data.k.dtype == jnp.float32 else jnp.complex128
    imag_unit = jnp.asarray(1.0j, dtype=complex_dtype)
    applied = applied_r.astype(complex_dtype) + imag_unit * applied_i.astype(complex_dtype)
    return applied.reshape(expected_shape)


def k_quadratic(
    trial_data: CisdKTrial,
    matrix: jax.Array,
    applied: jax.Array | None = None,
) -> jax.Array:
    """Return the bilinear contraction ``matrix.T @ K @ matrix``."""
    if applied is None:
        applied = k_apply(trial_data, matrix)
    accumulator_dtype = (
        jnp.complex128 if jnp.issubdtype(matrix.dtype, jnp.complexfloating) else jnp.float64
    )
    return jnp.sum(matrix * applied, dtype=accumulator_dtype)


def overlap_r(walker: jax.Array, trial_data: CisdKTrial) -> jax.Array:
    """Exact overlap for a restricted walker and K-native CISD trial."""
    wocc = walker[: trial_data.nocc_full, :]
    green = jnp.linalg.solve(wocc.T, walker.T)
    det0 = jnp.linalg.det(wocc)

    green_occ = green[trial_data.occ_act_slice, trial_data.vir_act_slice]
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    doubles = k_quadratic(trial_data, green_occ)
    return (1.0 + 2.0 * ci1g + doubles) * det0 * det0


def make_cisd_k_trial_data(data: dict, sys: System) -> CisdKTrial:
    """Build an exact K-native trial from ``k`` or a temporary dense ``ci2``.

    The returned trial always stores ``ci1`` and K in float64. If ``ci2`` is
    supplied, it is used only to construct K and is not retained by the trial.
    """
    del sys

    ci1 = jnp.asarray(data["ci1"], dtype=jnp.float64)
    if "k" in data:
        kernel = jnp.asarray(data["k"], dtype=jnp.float64)
    elif "ci2" in data:
        kernel = build_spin_adapted_kernel(jnp.asarray(data["ci2"], dtype=jnp.float64))
    else:
        raise KeyError("K-native CISD trial data requires 'k' or 'ci2'.")

    nocc_t_core = int(jnp.asarray(data.get("nocc_t_core", 0)).item())
    nvir_t_outer = int(jnp.asarray(data.get("nvir_t_outer", 0)).item())
    return CisdKTrial(
        ci1=ci1,
        k=kernel,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )


def make_cisd_k_trial_ops(sys: System) -> TrialOps:
    """Build restricted, closed-shell trial operations for ``CisdKTrial``."""
    if sys.nup != sys.ndn:
        raise ValueError("Restricted CISD K trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD K trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
