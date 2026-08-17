from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessTrial:
    """
    Restricted first-order PT-CCSD trial with T1 absorbed into the reference.

    The Thouless-rotated reference determinant is stored as ``mo_t`` in the
    current MO basis.  If ``mo_t`` is built from CCSD singles, its columns are
    ``[I; t1.T]``.  The ``t2`` tensor is the unmodified CCSD doubles amplitude
    in the original HF occupied/virtual partition, with shape
    ``(nocc, nvir, nocc, nvir)``.
    """

    mo_t: jax.Array
    t2: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.mo_t, "ndim") or not hasattr(self.t2, "ndim"):
            return
        if self.mo_t.ndim != 2:
            raise ValueError(
                f"PtccsdThoulessTrial.mo_t must be rank 2, got shape {self.mo_t.shape}."
            )
        if self.t2.ndim != 4:
            raise ValueError(f"PtccsdThoulessTrial.t2 must be rank 4, got shape {self.t2.shape}.")
        nocc, nvir = self.t2.shape[:2]
        if self.t2.shape != (nocc, nvir, nocc, nvir):
            raise ValueError(
                "PtccsdThoulessTrial.t2 must have shape "
                f"(nocc, nvir, nocc, nvir); got {self.t2.shape}."
            )
        if self.mo_t.shape != (nocc + nvir, nocc):
            raise ValueError(
                "PtccsdThoulessTrial.mo_t must have shape (nocc+nvir, nocc); "
                f"got {self.mo_t.shape} for t2 shape {self.t2.shape}."
            )

    @property
    def nocc(self) -> int:
        return int(self.t2.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t2.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        return (self.mo_t, self.t2), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t, t2 = children
        return cls(mo_t=mo_t, t2=t2)


def thouless_mo_from_t1(t1: jax.Array) -> jax.Array:
    """Return the occupied orbital matrix for ``exp(T1)|HF>`` in the MO basis."""
    nocc, nvir = t1.shape
    eye = jnp.eye(nocc, dtype=t1.dtype)
    return jnp.vstack([eye, t1.T])


def make_ptccsd_thouless_trial_data(data: dict) -> PtccsdThoulessTrial:
    t2 = jnp.asarray(data["t2"])
    if "mo_t" in data:
        mo_t = jnp.asarray(data["mo_t"])
    else:
        mo_t = thouless_mo_from_t1(jnp.asarray(data["t1"]))
    return PtccsdThoulessTrial(mo_t=mo_t, t2=t2)


def get_rdm1(trial_data: PtccsdThoulessTrial) -> jax.Array:
    c = trial_data.mo_t
    s = c.conj().T @ c
    dm = c @ jnp.linalg.solve(s, c.conj().T)
    return jnp.stack([dm, dm], axis=0)


def det_overlap_r(walker: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    return jnp.linalg.det(trial_data.mo_t.conj().T @ walker) ** 2


def half_green_restricted(walker: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    """
    Return ``(walker @ inv(<mo_t|walker>)).T`` with shape ``(nocc, norb)``.
    """
    overlap_mat = trial_data.mo_t.conj().T @ walker
    return jnp.linalg.solve(overlap_mat.T, walker.T)


def greens_restricted(walker: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    """
    Return the spin-block transition density transpose, ``P.T``.

    ``P = |walker> inv(<mo_t|walker>) <mo_t|``.  For an unrotated HF reference,
    the occupied rows reduce to the usual half-rotated Green's function and
    the virtual rows are zero.
    """
    return trial_data.mo_t.conj() @ half_green_restricted(walker, trial_data)


def greenp_from_green(green: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    return (green - jnp.eye(trial_data.norb, dtype=green.dtype))[:, trial_data.nocc :]


def theta_doubles(t2: jax.Array, green_occ: jax.Array) -> jax.Array:
    direct = jnp.einsum("iajb,ia,jb->", t2, green_occ, green_occ, optimize="optimal")
    exchange = jnp.einsum("iajb,ib,ja->", t2, green_occ, green_occ, optimize="optimal")
    return 2.0 * direct - exchange


def theta_t2_from_green(green: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    green_occ = green[: trial_data.nocc, trial_data.nocc :]
    return theta_doubles(trial_data.t2, green_occ)


def theta_t2_r(walker: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    return theta_t2_from_green(greens_restricted(walker, trial_data), trial_data)


def overlap_ptccsd_thouless_r(walker: jax.Array, trial_data: PtccsdThoulessTrial) -> jax.Array:
    return det_overlap_r(walker, trial_data) * jnp.exp(theta_t2_r(walker, trial_data))


def make_ptccsd_thouless_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("PT-CCSD Thouless trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless trial currently supports only restricted walkers, "
            f"got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_ptccsd_thouless_r, get_rdm1=get_rdm1)
