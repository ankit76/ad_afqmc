"""Abstract tracing on CPU; numerical CISD/UCISD checks execute only on GPUs."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from trot import walkers as wk
from trot.core.ops import d_energy_sampling_noise, d_energy_head_guard_count
from trot.ham.chol import HamChol
from trot.meas import cisd_modes as cm, ucisd_k_modes as um
from trot.meas.cisd import CisdMeasCfg
from trot.meas.ucisd import UcisdMeasCfg
from trot.meas.pair_sampling import local_cholesky_head, with_local_cholesky_sampling
from trot.sharding import plan_cholesky_layout, replicate, shard_cholesky_layout
from trot.trial.cisd_modes import CisdModeTrial
from trot.trial.ucisd_k_modes import UcisdKModeTrial

jax.config.update("jax_enable_x64", True)


def host_inputs(kind, mixed):
    """Small symmetry-correct amplitudes, prepared with NumPy only."""
    rng = np.random.default_rng(441)
    dtype = np.float32 if mixed else np.float64
    if kind == "cisd":
        raw = rng.normal(size=(6, 6)); raw = .01 * (raw + raw.T)
        c2 = raw.reshape(2, 3, 2, 3)
        kernel = (2*c2 - c2.transpose(0, 3, 2, 1)).reshape(6, 6)
        vals, vecs = np.linalg.eigh(kernel)
        keep = np.argsort(abs(vals))[-5:]
        trial = CisdModeTrial(.03*rng.normal(size=(2, 3)), vals[keep],
            vecs[:, keep].T.reshape(5, 2, 3).astype(dtype), nocc_t_core=1, nvir_t_outer=1)
        n, no = 7, 3
    else:
        n, no = 5, 2
        def same_spin(no, nv):
            a = rng.normal(size=(no, nv, no, nv))
            return .005*(a-a.transpose(2, 1, 0, 3)-a.transpose(0, 3, 2, 1)+a.transpose(2, 3, 0, 1))
        aa, bb = same_spin(2, 3).reshape(6, 6), same_spin(1, 4).reshape(4, 4)
        ab = .02*rng.normal(size=(6, 4))
        vals, vecs = np.linalg.eigh(np.block([[aa, ab], [ab.T, bb]]))
        keep = np.argsort(abs(vals))[-5:]
        rotation, _ = np.linalg.qr(np.eye(n)+.15*rng.normal(size=(n, n)))
        trial = UcisdKModeTrial(np.eye(n), rotation, .03*rng.normal(size=(2, 3)),
            .03*rng.normal(size=(1, 4)), vals[keep], vecs[:, keep].T.astype(dtype))
    chol = .01*rng.normal(size=(7, n, n)); chol = (chol+chol.transpose(0, 2, 1))/2
    h = HamChol(np.asarray(.2), np.diag(np.linspace(-1., 1., n)), chol)
    w = np.eye(n, no)[None]+.025*(rng.normal(size=(12, n, no))+1j*rng.normal(size=(12, n, no)))
    weights = np.arange(1, 13, dtype=float)
    return h, trial, w, weights


def context_cfg(kind, mixed):
    cls = CisdMeasCfg if kind == "cisd" else UcisdMeasCfg
    return cls(memory_mode="high", mixed_real_dtype=jnp.float32 if mixed else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed else jnp.complex128)


def common_fn(module):
    if module is cm:
        return cm._cisd_mode_energy_common
    return lambda w, h, c, t: um._ucisd_k_energy_common(w, h, c, t, um._k_mode_apply_realimag)


def prepare(kind, mixed, n_model, head_size, *, abstract=False):
    if jax.local_device_count() < n_model:
        pytest.skip(f"Requires {n_model} devices")
    module = cm if kind == "cisd" else um
    mesh = Mesh(np.asarray(jax.local_devices()[:n_model]).reshape(1, n_model), ("data", "model"),
                axis_types=(AxisType.Auto, AxisType.Auto))
    h, t, w, weights = host_inputs(kind, mixed)
    # Non-prefix head and nondivisible vector count exercise physical remapping.
    head = np.array([1, 4, 0, 6, 3, 5, 2])[:head_size]
    tail = np.setdiff1d(np.arange(7), head)
    p = np.arange(1, tail.size+1, dtype=float)
    p = p/p.sum() if p.size else p
    layout = plan_cholesky_layout(7, n_model, head, tail, p)
    h = HamChol(replicate(h.h0, mesh), replicate(h.h1, mesh), shard_cholesky_layout(h.chol, mesh, layout))
    t, w, weights = jax.tree.map(lambda a: replicate(a, mesh), (t, w, weights))
    cfg = context_cfg(kind, mixed)
    build = lambda h, t: module.build_meas_ctx(h, t, cfg=cfg, n_mode_chunks=3)
    if abstract and kind == "cisd":
        # The setup helper has top-level-only compiler options; its output
        # shapes suffice to trace the measurement without running setup.
        empty = jax.ShapeDtypeStruct((0,), np.float64)
        ctx = cm.CisdModeMeasCtx(
            jax.ShapeDtypeStruct((h.nchol, t.nocc_full, t.norb), np.float64),
            jax.ShapeDtypeStruct((h.nchol, t.norb, t.ci1.shape[0]), np.float64),
            empty, empty, empty, empty, cfg, 3, None, setup_mesh=mesh if n_model > 1 else None)
    else:
        ctx = jax.eval_shape(build, h, t) if abstract else build(h, t)
    sampling_cls = cm.CisdModePairSamplingCfg if kind == "cisd" else um.UcisdKModePairSamplingCfg
    ctx = replace(ctx, energy_sampling=sampling_cls(head_size, 17, head_chol_batch_size=2,
        guard_head_deviations=True, walker_guide_policy="head_rms", track_half_sample_diagnostic=True))
    ctx = with_local_cholesky_sampling(ctx, layout, mesh)
    return module, mesh, layout, h, ctx, t, w, weights


@pytest.mark.parametrize("kind", ["cisd", "ucisd"])
@pytest.mark.parametrize("mixed", [False, True])
def test_abstract_local_energy_trace(kind, mixed):
    # Traces the real contractions and their nested scans, without compiling or
    # executing an AFQMC energy kernel on the CPU.
    module, mesh, layout, h, ctx, t, w, weights = prepare(kind, mixed, 4, 3, abstract=True)
    args = (w, weights, np.ones(12, dtype=complex), np.array([0, 83], dtype=np.uint32),
            h, ctx, t, np.asarray(-2.), np.asarray(100.))
    for chunks in (1, 5):
        value = jax.eval_shape(lambda w, wt, ov, key, h, c, t, ref, clip:
            module.pair_sampled_block_energy(w, wt, ov, key, chunks, h, c, t, ref, clip), *args)
        assert value.energy.shape == ()
        assert value.energy.dtype == np.float64
    configure = cm.configure_cisd_mode_pair_sampling if kind == "cisd" else um.configure_ucisd_k_mode_pair_sampling
    with pytest.raises(ValueError, match="frozen guide"):
        configure(ctx, ctx.energy_sampling, np.ones(8))


@pytest.mark.parametrize("kind", ["cisd", "ucisd"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("n_model,head_size", [(1, 3), (4, 3), (4, 0), (4, 7)])
def test_gpu_local_energy_matches_explicit_sample_sum(kind, mixed, n_model, head_size):
    if jax.default_backend() != "gpu":
        pytest.skip("Numerical AFQMC kernel checks require GPUs")
    module, mesh, layout, h, ctx, t, w, weights = prepare(kind, mixed, n_model, head_size)
    # The reference uses global contractions and independently assembles the
    # stratum estimator from its known sampled indices, including guarded weights.
    common = jax.jit(wk.vmap_chunked(common_fn(module), 1, in_axes=(0, None, None, None)))(w, h, ctx, t)
    all_terms = jax.jit(lambda common, h, c, t: module._local_cholesky_head_terms(
        common, jnp.arange(h.chol.shape[0]), h, c, t, n_chunks=1))(common, h, ctx, t)
    terms = np.asarray(all_terms)
    heads = terms[:, layout.head_indices]
    expected_head, expected_squared = heads.sum(axis=1), (heads.real**2).sum(axis=1)
    for chunks in (1, 5):
        actual_head, actual_squared = jax.jit(lambda common, h, c, t: local_cholesky_head(
            common, h, c, t, terms_fn=module._local_cholesky_head_terms,
            context_specs_fn=module._local_cholesky_context_specs, n_chunks=chunks))(common, h, ctx, t)
        tol = 1e-6 if mixed else 1e-12
        np.testing.assert_allclose(actual_head, expected_head, rtol=tol, atol=tol)
        np.testing.assert_allclose(actual_squared, expected_squared, rtol=tol, atol=tol)

    pi = np.asarray(weights)/np.asarray(weights).sum()
    head_energy = np.asarray(common.base).real + expected_head.real
    deviations = abs(head_energy-np.sum(pi*head_energy))
    key = jax.random.PRNGKey(83)
    for clip in (100., float(np.median(deviations)), -1.):  # none, some, all guarded
        guarded = deviations > clip
        accepted = np.where(guarded, 0., pi)
        exact = np.sum(pi*np.where(guarded, -2., head_energy))
        noise = 0.
        if accepted.sum() and layout.tail_indices.size:
            q0 = accepted/accepted.sum()
            guide = accepted*np.sqrt(expected_squared)
            q = .1*q0+.9*(guide/guide.sum() if guide.sum() else q0)
            count, rem = divmod(17, n_model)
            capacity = count+bool(rem)
            width = layout.local_tail_prob.shape[1]
            for d, p in enumerate(layout.local_tail_prob):
                if p.sum() == 0:
                    continue
                p = p/p.sum()
                kw, kg = jax.random.split(jax.random.fold_in(key, d))
                uw, ug = (np.asarray(jax.random.uniform(k, (capacity,), dtype=jnp.float64)) for k in (kw, kg))
                iw = np.searchsorted(np.cumsum(q), uw*np.sum(q), side="right")
                ig = np.searchsorted(np.cumsum(p), ug*np.sum(p), side="right")
                vals = accepted[iw]*terms[iw, d*width+ig].real/(q[iw]*p[ig])
                vals = vals[:count+(d < rem)]
                exact += vals.mean()
                first = len(vals)//2; second = len(vals)-first
                noise += np.sqrt(first*second)/len(vals)*(vals[:first].mean()-vals[first:].mean())
        for chunks in (1, 5):
            f = jax.jit(lambda w, wt, h, c, t, key, clip: module.pair_sampled_block_energy(
                w, wt, jnp.ones(12, dtype=complex), key, chunks, h, c, t, jnp.asarray(-2.), clip))
            result = f(w, weights, h, ctx, t, key, jnp.asarray(clip))
            np.testing.assert_allclose(result.energy, exact, rtol=tol, atol=tol)
            np.testing.assert_allclose(result.diagnostics[d_energy_sampling_noise], noise, rtol=tol, atol=tol)
            assert result.diagnostics[d_energy_head_guard_count] == guarded.sum()
