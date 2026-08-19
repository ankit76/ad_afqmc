import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import (
    _build_prop_ctx,
    _make_vhs_split_flat,
    _packed_upper_size,
    make_trotter_ops,
)


def _make_small_ham(*, norb=4, n_fields=3, h0=0.0, seed=0):
    key = jax.random.PRNGKey(seed)

    a = jax.random.normal(key, (norb, norb))
    h1 = 0.1 * (a + a.T)

    key, sub = jax.random.split(key)
    chol = 0.05 * jax.random.normal(sub, (n_fields, norb, norb))
    chol = 0.5 * (chol + jnp.swapaxes(chol, -1, -2))

    return HamChol(basis="restricted", h0=jnp.asarray(h0), h1=h1, chol=chol)


def test_build_prop_ctx_shapes_and_nfields():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)

    dm = jnp.zeros((norb, norb))
    dt = 0.2
    ctx = _build_prop_ctx(ham, dm, dt)

    assert ctx.mf_shifts.shape == (n_fields,)
    assert ctx.exp_h1_half.shape == (norb, norb)
    assert ctx.dt.shape == ()
    assert ctx.sqrt_dt.shape == ()
    assert ctx.h0_prop.shape == ()


@pytest.mark.parametrize(
    ("chol_dtype", "complex_dtype", "rtol"),
    [
        (jnp.float32, jnp.complex64, 2.0e-6),
        (jnp.float64, jnp.complex128, 1.0e-12),
    ],
)
def test_packed_cholesky_reconstructs_full_vhs(chol_dtype, complex_dtype, rtol):
    norb, n_fields = 6, 9
    ham = _make_small_ham(norb=norb, n_fields=n_fields, seed=17)
    dm = jnp.zeros((norb, norb), dtype=jnp.float64)
    field = jax.random.normal(jax.random.PRNGKey(31), (n_fields,), dtype=jnp.float64)
    field = field + 1.0j * jax.random.normal(
        jax.random.PRNGKey(32), (n_fields,), dtype=jnp.float64
    )

    full_ctx = _build_prop_ctx(
        ham,
        dm,
        0.01,
        chol_flat_precision=chol_dtype,
        packed_cholesky=False,
    )
    packed_ctx = _build_prop_ctx(
        ham,
        dm,
        0.01,
        chol_flat_precision=chol_dtype,
        packed_cholesky=True,
    )

    assert full_ctx.chol_flat.shape == (n_fields, norb * norb)
    assert packed_ctx.chol_flat.shape == (n_fields, _packed_upper_size(norb))
    assert not full_ctx.chol_packed
    assert packed_ctx.chol_packed

    full_vhs = jax.jit(
        lambda x: _make_vhs_split_flat(
            chol_flat=full_ctx.chol_flat,
            x=x,
            n=norb,
        )
    )(field.astype(complex_dtype))
    packed_vhs = jax.jit(
        lambda x: _make_vhs_split_flat(
            chol_flat=packed_ctx.chol_flat,
            x=x,
            n=norb,
            chol_packed=True,
        )
    )(field.astype(complex_dtype))

    np.testing.assert_allclose(packed_vhs, full_vhs, rtol=rtol, atol=rtol)
    np.testing.assert_allclose(packed_vhs, packed_vhs.T, rtol=0.0, atol=0.0)


def test_packed_and_full_trotter_actions_match():
    norb, nocc, n_fields = 5, 2, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, seed=23)
    dm = jnp.zeros((norb, norb), dtype=jnp.float64)
    walker = jax.random.normal(
        jax.random.PRNGKey(41), (norb, nocc), dtype=jnp.float64
    ).astype(jnp.complex128)
    field = jax.random.normal(jax.random.PRNGKey(42), (n_fields,), dtype=jnp.float64)
    field = field + 1.0j * jax.random.normal(
        jax.random.PRNGKey(43), (n_fields,), dtype=jnp.float64
    )

    full_ctx = _build_prop_ctx(ham, dm, 0.005, packed_cholesky=False)
    packed_ctx = _build_prop_ctx(ham, dm, 0.005, packed_cholesky=True)
    trotter_ops = make_trotter_ops("restricted", "restricted", mixed_precision=False)

    full_walker = trotter_ops.apply_trotter(walker, field, full_ctx, 6)
    packed_walker = jax.jit(trotter_ops.apply_trotter, static_argnums=3)(
        walker, field, packed_ctx, 6
    )

    np.testing.assert_allclose(packed_walker, full_walker, rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    pytest.main([__file__])
