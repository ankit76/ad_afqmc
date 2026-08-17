from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdTrial:
    """
    Restricted first-order PT-CCSD trial in the canonical HF MO basis.

    Arrays:
      t1: (nocc, nvir)
      t2: (nocc, nvir, nocc, nvir)

    ``t2`` is the raw CCSD doubles amplitude. It is not the CISD-converted
    coefficient ``T2 + T1^2``.
    """

    t1: jax.Array
    t2: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.t1, "ndim") or not hasattr(self.t2, "ndim"):
            return
        if self.t1.ndim != 2:
            raise ValueError(f"PtccsdTrial.t1 must be rank 2, got shape {self.t1.shape}.")
        if self.t2.ndim != 4:
            raise ValueError(f"PtccsdTrial.t2 must be rank 4, got shape {self.t2.shape}.")
        nocc, nvir = self.t1.shape
        if self.t2.shape != (nocc, nvir, nocc, nvir):
            raise ValueError(
                "PtccsdTrial.t2 must have shape (nocc, nvir, nocc, nvir); "
                f"got {self.t2.shape} for t1 shape {self.t1.shape}."
            )

    @property
    def nocc(self) -> int:
        return int(self.t1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t1.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        return (self.t1, self.t2), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        t1, t2 = children
        return cls(t1=t1, t2=t2)


def make_ptccsd_trial_data(data: dict) -> PtccsdTrial:
    return PtccsdTrial(t1=jnp.asarray(data["t1"]), t2=jnp.asarray(data["t2"]))


def get_rdm1(trial_data: PtccsdTrial) -> jax.Array:
    occ = jnp.arange(trial_data.norb) < trial_data.nocc
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


def greens_restricted(walker: jax.Array, trial_data: PtccsdTrial) -> jax.Array:
    """
    Half-rotated Green's function for a restricted walker in the HF MO basis.

    Returns an array with shape ``(nocc, norb)``.
    """
    wocc = walker[: trial_data.nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def hf_overlap_r(walker: jax.Array, trial_data: PtccsdTrial) -> jax.Array:
    wocc = walker[: trial_data.nocc, :]
    det0 = jnp.linalg.det(wocc)
    return det0 * det0


def theta_singles(t1: jax.Array, green_occ: jax.Array) -> jax.Array:
    return 2.0 * jnp.einsum("ia,ia->", t1, green_occ, optimize="optimal")


def theta_doubles(t2: jax.Array, green_occ: jax.Array) -> jax.Array:
    direct = jnp.einsum("iajb,ia,jb->", t2, green_occ, green_occ, optimize="optimal")
    exchange = jnp.einsum("iajb,ib,ja->", t2, green_occ, green_occ, optimize="optimal")
    return 2.0 * direct - exchange


def theta_pt_from_green(green: jax.Array, trial_data: PtccsdTrial) -> jax.Array:
    green_occ = green[:, trial_data.nocc :]
    return theta_singles(trial_data.t1, green_occ) + theta_doubles(trial_data.t2, green_occ)


def theta_pt_r(walker: jax.Array, trial_data: PtccsdTrial) -> jax.Array:
    return theta_pt_from_green(greens_restricted(walker, trial_data), trial_data)


def overlap_pt_r(walker: jax.Array, trial_data: PtccsdTrial) -> jax.Array:
    return hf_overlap_r(walker, trial_data) * jnp.exp(theta_pt_r(walker, trial_data))


def make_ptccsd_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("PT-CCSD trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"PT-CCSD trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_pt_r, get_rdm1=get_rdm1)
