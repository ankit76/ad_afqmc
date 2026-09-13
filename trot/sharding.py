from dataclasses import dataclass
from typing import TypeVar, cast

from numpy.typing import DTypeLike, NDArray

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .ham.chol import HamChol
from .ham.hubbard import HamHubbard
from .prop.types import PropState

THam = TypeVar("THam")
ArrayLike = jax.Array | NDArray[np.generic]


@dataclass(frozen=True)
class CholeskyLayout:
    """Host-side layout in original Cholesky identities; -1 denotes padding.

    Apply ``permutation`` to the Hamiltonian before constructing derived
    contexts. Head/tail indices below address the resulting global array.
    The local arrays describe equal-sized contiguous model shards.
    """

    permutation: NDArray[np.int64]
    inverse_permutation: NDArray[np.int64]
    head_indices: NDArray[np.int32]
    tail_indices: NDArray[np.int32]
    tail_prob: NDArray[np.float64]
    local_head_indices: NDArray[np.int32]
    local_head_valid: NDArray[np.bool_]
    local_tail_prob: NDArray[np.float64]

    @property
    def n_model(self) -> int:
        return int(self.local_tail_prob.shape[0])

    @property
    def n_chol(self) -> int:
        return int(self.inverse_permutation.size)


def plan_cholesky_layout(
    n_chol: int,
    n_model: int,
    head_indices: ArrayLike,
    tail_indices: ArrayLike,
    tail_prob: ArrayLike,
) -> CholeskyLayout:
    """Balance exact-head counts and tail probability mass on the host.

    Head membership and the proposal are supplied by the caller. Greedily
    assign descending tail probabilities to the least-loaded shard with space.
    Every real vector has one owner; padding receives no sampling probability.
    This deterministic planner does not select a guide or change the head.
    """
    if not isinstance(n_chol, (int, np.integer)) or n_chol <= 0:
        raise ValueError("n_chol must be a positive integer.")
    if not isinstance(n_model, (int, np.integer)) or n_model <= 0:
        raise ValueError("n_model must be a positive integer.")
    head, tail = np.asarray(head_indices), np.asarray(tail_indices)
    if any(a.ndim != 1 or (a.size and a.dtype.kind not in "iu") for a in (head, tail)):
        raise ValueError("Head and tail indices must be one-dimensional integer arrays.")
    head, tail = head.astype(np.int64), tail.astype(np.int64)
    if not np.array_equal(np.sort(np.concatenate((head, tail))), np.arange(n_chol)):
        raise ValueError("Head and tail must partition the original Cholesky indices exactly.")
    prob = np.asarray(tail_prob, dtype=np.float64)
    if prob.shape != tail.shape or not np.isfinite(prob).all() or np.any(prob <= 0):
        raise ValueError("Every tail vector needs a finite, positive probability.")
    if prob.size and not np.isclose(prob.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Tail probabilities must sum to one.")
    prob = prob.copy() / prob.sum() if prob.size else prob.copy()
    width = (n_chol + n_model - 1) // n_model
    groups = [list(head[d::n_model]) for d in range(n_model)]
    head_counts = np.array([len(g) for g in groups])
    remaining = width - head_counts
    mass = np.zeros(n_model)
    for i in np.argsort(-prob, kind="stable"):
        d = int(np.argmin(np.where(remaining > 0, mass, np.inf)))
        groups[d].append(int(tail[i]))
        remaining[d] -= 1
        mass[d] += prob[i]
    permutation = np.full((n_model, width), -1, dtype=np.int64)
    for d, group in enumerate(groups):
        permutation[d, :len(group)] = group
    permutation = permutation.ravel()
    inverse = np.empty(n_chol, dtype=np.int64)
    valid = permutation >= 0
    inverse[permutation[valid]] = np.flatnonzero(valid)
    head_new, tail_new = inverse[head].astype(np.int32), inverse[tail].astype(np.int32)
    hwidth = int(head_counts.max(initial=0))
    local_head = np.broadcast_to(np.arange(hwidth, dtype=np.int32), (n_model, hwidth)).copy()
    head_valid = local_head < head_counts[:, None]
    local_head[~head_valid] = 0
    local_prob = np.zeros(n_model * width, dtype=np.float64)
    local_prob[tail_new] = prob
    return CholeskyLayout(permutation, inverse, head_new, tail_new, prob,
                          local_head, head_valid, local_prob.reshape(n_model, width))


def shard_cholesky_layout(
    x: NDArray[np.generic], mesh: Mesh, layout: CholeskyLayout, *, dtype: DTypeLike | None = None,
) -> jax.Array:
    """Place a host array using the layout without a full permuted host copy.

    Use the same layout for all original-order Cholesky arrays, or build
    derived contexts from the reordered Hamiltonian. Does not mutate ``x``.
    """
    if not isinstance(x, np.ndarray) or x.ndim == 0 or x.shape[0] != layout.n_chol:
        raise ValueError("Expected an original-order NumPy array with layout.n_chol rows.")
    if not has_model_axis(mesh) or _mesh_axis_size(mesh, "model") != layout.n_model:
        raise ValueError("Mesh model size does not match the Cholesky layout.")
    target_dtype = x.dtype if dtype is None else np.dtype(dtype)

    def block(index):
        indices = layout.permutation[index[0]]
        out = np.zeros((indices.size, *x.shape[1:]), dtype=target_dtype)
        valid = indices >= 0
        out[valid] = x[indices[valid]]
        return out

    return jax.make_array_from_callback(
        (layout.permutation.size, *x.shape[1:]), NamedSharding(mesh, P("model")),
        block, dtype=target_dtype,
    )


def make_data_mesh() -> Mesh:
    n = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n,))
    return Mesh(devices, ("data",))


def make_data_model_mesh(n_data: int | None = None, n_model: int | None = None) -> Mesh:
    n = jax.local_device_count()

    if n_data is None and n_model is None:
        n_data, n_model = 1, n
    elif n_data is None:
        assert n_model is not None
        if n % n_model != 0:
            raise ValueError(f"local_device_count={n} is not divisible by n_model={n_model}.")
        n_data = n // n_model
    elif n_model is None:
        if n % n_data != 0:
            raise ValueError(f"local_device_count={n} is not divisible by n_data={n_data}.")
        n_model = n // n_data

    assert n_data is not None and n_model is not None
    if n_data * n_model != n:
        raise ValueError(
            f"Requested mesh ({n_data}, {n_model}) uses {n_data * n_model} devices, "
            f"but {n} local devices are visible."
        )

    devices = mesh_utils.create_device_mesh((n_data, n_model))
    return Mesh(devices, ("data", "model"))


def has_model_axis(mesh: Mesh | None) -> bool:
    return mesh is not None and "model" in mesh.axis_names


def cholesky_model_mesh(chol: ArrayLike) -> Mesh | None:
    """Inspect a concrete setup input for first-axis-only model sharding.

    Call before JIT tracing; Auto-axis tracers do not retain input placement.
    Pass the returned mesh explicitly to compiled setup helpers.
    """
    sharding = getattr(chol, "sharding", None)
    if not isinstance(sharding, NamedSharding) or not has_model_axis(sharding.mesh):
        return None
    spec = tuple(sharding.spec)
    if not spec or spec[0] != "model" or any(axis is not None for axis in spec[1:]):
        return None
    return sharding.mesh if _mesh_axis_size(sharding.mesh, "model") > 1 else None


def _mesh_axis_size(mesh: Mesh, axis_name: str) -> int:
    return dict(zip(mesh.axis_names, mesh.devices.shape, strict=True))[axis_name]


def _pad_for_model_axis(chol: ArrayLike, mesh: Mesh) -> ArrayLike:
    n_model = _mesh_axis_size(mesh, "model")
    n_chol = int(chol.shape[0])
    remainder = n_chol % n_model
    if remainder == 0:
        return chol

    padded_n_chol = n_chol + (n_model - remainder)
    pad = padded_n_chol - n_chol
    print(
        f"[shard] padding chol from {n_chol} to {padded_n_chol} "
        f"to shard evenly over n_model={n_model}.",
        flush=True,
    )

    pad_width = [(0, 0)] * chol.ndim
    pad_width[0] = (0, pad)
    if isinstance(chol, np.ndarray):
        return cast(ArrayLike, np.pad(chol, pad_width, mode="constant"))
    return cast(ArrayLike, jnp.pad(chol, pad_width, mode="constant"))


def shard_first_axis(x: ArrayLike, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P("data")))


def shard_model_axis(
    x: ArrayLike,
    mesh: Mesh,
    *,
    dtype: DTypeLike | None = None,
    announce_padding: bool = True,
) -> jax.Array:
    sharding = NamedSharding(mesh, P("model"))
    target_dtype = np.dtype(dtype) if dtype is not None else None

    if isinstance(x, np.ndarray):
        n_model = _mesh_axis_size(mesh, "model")
        n_chol = int(x.shape[0])
        remainder = n_chol % n_model
        padded_n_chol = n_chol + (n_model - remainder) if remainder != 0 else n_chol
        needs_cast = target_dtype is not None and x.dtype != target_dtype
        if remainder != 0 or needs_cast:
            if remainder != 0 and announce_padding:
                print(
                    f"[shard] padding chol from {n_chol} to {padded_n_chol} "
                    f"to shard evenly over n_model={n_model}.",
                    flush=True,
                )

            out_dtype = x.dtype if target_dtype is None else target_dtype

            def _callback(index):
                if index is None:
                    raise ValueError("addressable shard index unexpectedly None")
                head = index[0]
                assert isinstance(head, slice)
                start = 0 if head.start is None else int(head.start)
                # A size-one model axis is replicated; JAX can request slice(None).
                stop = padded_n_chol if head.stop is None else int(head.stop)
                if stop <= n_chol and target_dtype is None:
                    return x[index]

                shard = np.zeros((stop - start, *x.shape[1:]), dtype=out_dtype)
                valid_stop = min(stop, n_chol)
                if valid_stop > start:
                    shard[: valid_stop - start] = np.asarray(
                        x[start:valid_stop],
                        dtype=out_dtype,
                    )
                return shard

            return jax.make_array_from_callback(
                (padded_n_chol, *x.shape[1:]),
                sharding,
                _callback,
                dtype=out_dtype,  # type: ignore
            )

    n_model = _mesh_axis_size(mesh, "model")
    if int(x.shape[0]) % n_model != 0:
        if announce_padding:
            x = _pad_for_model_axis(x, mesh)
        else:
            remainder = int(x.shape[0]) % n_model
            pad_width = [(0, 0)] * x.ndim
            pad_width[0] = (0, n_model - remainder)
            x = cast(ArrayLike, jnp.pad(x, pad_width, mode="constant"))

    if target_dtype is not None:
        x = cast(ArrayLike, jnp.asarray(x, dtype=target_dtype))
    return jax.device_put(x, sharding)


def replicate(x: ArrayLike, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P()))


def shard_ham_data(ham_data: THam, mesh: Mesh | None) -> THam:
    """
    For a data x model mesh:
      - replicate h0/h1
      - shard chol on the model axis
    """
    if mesh is None or mesh.size == 1 or not has_model_axis(mesh):
        return ham_data

    if isinstance(ham_data, HamChol):
        nchol = ham_data.nchol if int(ham_data.chol.shape[0]) == 0 else None
        return cast(
            THam,
            HamChol(
                h0=replicate(ham_data.h0, mesh),
                h1=replicate(ham_data.h1, mesh),
                chol=shard_model_axis(ham_data.chol, mesh),
                basis=ham_data.basis,
                nchol=nchol,
            ),
        )

    if isinstance(ham_data, HamHubbard):
        raise ValueError("Cannot shard Hubbard Hamiltonian, don't use model axis sharding.")

    return ham_data


def shard_prop_state(state: PropState, mesh: Mesh | None) -> PropState:
    """
    Shard only (n_walkers,...) leaves, keep global scalars replicated.
    """
    if mesh is None or mesh.size == 1:
        return state

    walkers_sh = tree_util.tree_map(lambda a: shard_first_axis(a, mesh), state.walkers)

    return state._replace(
        walkers=walkers_sh,
        weights=shard_first_axis(state.weights, mesh),
        overlaps=shard_first_axis(state.overlaps, mesh),
        rng_key=replicate(state.rng_key, mesh),
        pop_control_ene_shift=replicate(state.pop_control_ene_shift, mesh),
        e_estimate=replicate(state.e_estimate, mesh),
        node_encounters=replicate(state.node_encounters, mesh),
    )
