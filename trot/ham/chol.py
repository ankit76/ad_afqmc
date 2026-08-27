from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util

HamBasis = Literal["restricted", "unrestricted", "generalized"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamChol:
    """
    cholesky hamiltonian.

    basis="restricted":
      h1:   (norb, norb)
      chol: (n_fields, norb, norb)

    basis="generalized":
      h1:   (nso, nso)   where nso = 2*norb
      chol: (n_fields, nso, nso)
    """

    h0: jax.Array
    h1: jax.Array
    chol: jax.Array
    basis: HamBasis = "restricted"
    nchol: int | None = None

    def __post_init__(self):
        if self.basis not in ("restricted", "generalized"):
            raise ValueError(f"unknown basis: {self.basis}")
        chol_shape = getattr(self.chol, "shape", None)
        if chol_shape is None:
            return

        n_chol_shape = int(chol_shape[0])
        nchol = self.nchol
        if nchol is None:
            object.__setattr__(self, "nchol", n_chol_shape)
        elif n_chol_shape not in (0, int(nchol)):
            raise ValueError(f"nchol={nchol} is inconsistent with chol.shape[0]={n_chol_shape}")

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol)
        nchol = self.nchol
        assert nchol is not None
        aux = (self.basis, int(nchol))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol = children
        basis, nchol = aux
        return cls(h0=h0, h1=h1, chol=chol, basis=basis, nchol=nchol)


def n_fields(ham: HamChol | HamCholUhf) -> int:
    nchol = ham.nchol
    assert nchol is not None
    return int(nchol)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamCholUhf:
    """Spin-resolved Cholesky Hamiltonian with shared auxiliary fields.

    The alpha and beta one-particle bases may represent different orbital
    spaces.  At runtime both tensors use the same padded orbital dimension so
    that the existing unrestricted walker layout remains unchanged.  Padding
    rows and columns must be identically zero.

    ``chol_a[g]`` and ``chol_b[g]`` are the two actions of one and the same
    auxiliary field ``g``; they must never be sampled independently.
    """

    h0: jax.Array
    h1_a: jax.Array
    h1_b: jax.Array
    chol_a: jax.Array
    chol_b: jax.Array
    nchol: int | None = None
    norb_spin: tuple[int, int] | None = None
    basis: Literal["unrestricted"] = "unrestricted"

    def __post_init__(self):
        if self.basis != "unrestricted":
            raise ValueError(f"HamCholUhf basis must be 'unrestricted', got {self.basis!r}.")

        shapes = [
            getattr(x, "shape", None)
            for x in (self.h1_a, self.h1_b, self.chol_a, self.chol_b)
        ]
        if any(shape is None for shape in shapes):
            return
        h1_a_shape, h1_b_shape, chol_a_shape, chol_b_shape = shapes

        if len(h1_a_shape) != 2 or h1_a_shape[0] != h1_a_shape[1]:
            raise ValueError(f"h1_a must be square, got shape {h1_a_shape}.")
        if h1_b_shape != h1_a_shape:
            raise ValueError(
                "padded alpha and beta one-body tensors must have identical shapes, "
                f"got {h1_a_shape} and {h1_b_shape}."
            )
        padded_norb = int(h1_a_shape[0])
        if self.norb_spin is None:
            object.__setattr__(self, "norb_spin", (padded_norb, padded_norb))
        else:
            norb_a, norb_b = (int(self.norb_spin[0]), int(self.norb_spin[1]))
            if norb_a <= 0 or norb_b <= 0 or max(norb_a, norb_b) != padded_norb:
                raise ValueError(
                    "norb_spin must contain positive native dimensions whose maximum "
                    f"equals the padded size {padded_norb}, got {self.norb_spin}."
                )
        expected_tail = h1_a_shape
        if len(chol_a_shape) != 3 or chol_a_shape[1:] not in (expected_tail, (0, 0)):
            raise ValueError(
                f"chol_a must end in shape {expected_tail} (or be compact), got {chol_a_shape}."
            )
        if len(chol_b_shape) != 3 or chol_b_shape[1:] not in (expected_tail, (0, 0)):
            raise ValueError(
                f"chol_b must end in shape {expected_tail} (or be compact), got {chol_b_shape}."
            )
        if chol_a_shape[0] != chol_b_shape[0]:
            raise ValueError(
                "alpha and beta Cholesky tensors must have the same field dimension, "
                f"got {chol_a_shape[0]} and {chol_b_shape[0]}."
            )

        n_chol_shape = int(chol_a_shape[0])
        nchol = self.nchol
        if nchol is None:
            object.__setattr__(self, "nchol", n_chol_shape)
        elif n_chol_shape not in (0, int(nchol)):
            raise ValueError(
                f"nchol={nchol} is inconsistent with Cholesky field dimension={n_chol_shape}."
            )

    @property
    def h1(self) -> jax.Array:
        """Alpha tensor alias for code whose spin-beta tensor lives in a context."""
        return self.h1_a

    @property
    def chol(self) -> jax.Array:
        """Alpha tensor alias; the beta tensor is always ``chol_b``."""
        return self.chol_a

    def tree_flatten(self):
        children = (self.h0, self.h1_a, self.h1_b, self.chol_a, self.chol_b)
        nchol = self.nchol
        assert nchol is not None
        norb_spin = self.norb_spin
        assert norb_spin is not None
        return children, (int(nchol), tuple(int(n) for n in norb_spin))

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1_a, h1_b, chol_a, chol_b = children
        nchol, norb_spin = aux
        return cls(
            h0=h0,
            h1_a=h1_a,
            h1_b=h1_b,
            chol_a=chol_a,
            chol_b=chol_b,
            nchol=int(nchol),
            norb_spin=norb_spin,
        )


HamCholData = HamChol | HamCholUhf


def slice_ham_level(
    ham: HamCholData,
    *,
    norb_keep: int | None,
    nchol_keep: int | None,
) -> HamCholData:
    """
    Build a HamChol view for measurement in MLMC:
      - slice orbitals as a prefix [:norb_keep]
      - slice chol as a prefix [:nchol_keep]
    """
    h0 = ham.h0
    new_nchol = ham.nchol

    if isinstance(ham, HamCholUhf):
        h1_a = ham.h1_a
        h1_b = ham.h1_b
        chol_a = ham.chol_a
        chol_b = ham.chol_b

        if norb_keep is not None:
            h1_a = h1_a[:norb_keep, :norb_keep]
            h1_b = h1_b[:norb_keep, :norb_keep]
            chol_a = chol_a[:, :norb_keep, :norb_keep]
            chol_b = chol_b[:, :norb_keep, :norb_keep]

        if nchol_keep is not None:
            chol_a = chol_a[:nchol_keep]
            chol_b = chol_b[:nchol_keep]
            assert new_nchol is not None
            new_nchol = min(int(new_nchol), nchol_keep)

        return HamCholUhf(
            h0=h0,
            h1_a=h1_a,
            h1_b=h1_b,
            chol_a=chol_a,
            chol_b=chol_b,
            nchol=new_nchol,
            norb_spin=(
                (min(ham.norb_spin[0], norb_keep), min(ham.norb_spin[1], norb_keep))
                if norb_keep is not None and ham.norb_spin is not None
                else ham.norb_spin
            ),
        )

    h1 = ham.h1
    chol = ham.chol

    if norb_keep is not None:
        h1 = h1[:norb_keep, :norb_keep]
        chol = chol[:, :norb_keep, :norb_keep]

    if nchol_keep is not None:
        chol = chol[:nchol_keep]
        ham_nchol = ham.nchol
        assert ham_nchol is not None
        new_nchol = min(int(ham_nchol), nchol_keep)

    return HamChol(h0=h0, h1=h1, chol=chol, basis=ham.basis, nchol=new_nchol)
