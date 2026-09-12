"""Numerical and communication checks for sampling on local walker shards."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from dataclasses import dataclass, replace
from functools import partial
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from trot.core.ops import d_energy_sampling_noise
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas import cisd_modes as cm, ucisd_k_modes as um
from trot.meas.pair_sampling import _sample_indices, local_pair_mesh, local_pair_tail
from trot.sharding import replicate, shard_first_axis

jax.config.update("jax_enable_x64", True)


def mesh4():
    if jax.local_device_count() < 4:
        pytest.skip("Requires four logical devices")
    return Mesh(np.asarray(jax.local_devices()[:4]).reshape(4, 1), ("data", "model"),
                axis_types=(AxisType.Auto, AxisType.Auto))


def assert_scalar_collectives(compiled):
    for line in compiled.as_text().splitlines():
        if re.search(r"\b(all-gather|all-to-all|collective-permute)(-start|-done)?\(", line):
            pytest.fail(f"Unexpected data movement: {line}")
        if re.search(r"\b(all-reduce|reduce-scatter)(-start|-done)?\(", line):
            shapes = re.findall(r"(?:f|s|u|c)\d+\[([\d,]*)\]", line.split("metadata=")[0])
            assert all(np.prod([int(d) for d in s.split(',') if d]) <= 8 for s in shapes), line


@partial(jax.tree_util.register_dataclass,
         data_fields=["chol_tail_indices", "chol_tail_prob"], meta_fields=["energy_sampling"])
@dataclass(frozen=True)
class TableContext:
    chol_tail_indices: object
    chol_tail_prob: object
    energy_sampling: object


def table_terms(table, iw, ig, h, c, t, *, n_chunks):
    return table[iw, ig]


@pytest.mark.parametrize("p", [[1.], [0., .2, 0., .8, 0.], [.1, .2, .3, .4]])
def test_local_categorical_draw_matches_inverse_cdf(p):
    key = jax.random.PRNGKey(94)
    cdf = np.cumsum(p)
    targets = np.asarray(jax.random.uniform(key, (1000,), dtype=jnp.float64)) * cdf[-1]
    expected = np.searchsorted(cdf, targets, side="right")
    actual = jax.jit(_sample_indices, static_argnums=2)(key, jnp.asarray(p), 1000)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.asarray(p)[actual] > 0)


@pytest.mark.parametrize("empty_shard", [False, True])
def test_stratified_mean_variance_and_half_sample_noise(empty_shard):
    mesh = mesh4()
    table = np.arange(24, dtype=float).reshape(8, 3) / 9 - 1
    pi = np.array([1, 2, 3, 5, 8, 13, 21, 34], dtype=float)
    pi /= pi.sum()
    if empty_shard:
        pi[-2:] = 0  # Retain original global weight, as for guarded walkers.
        table[-2:] = np.nan  # Must not evaluate an empty shard's invalid walkers.
    q = pi * np.array([1, 4, 2, 1, 5, 2, 3, 1])
    q /= q.sum()
    p = np.array([.2, .3, .5])
    sampling = cm.CisdModePairSamplingCfg(0, 17, track_half_sample_diagnostic=True,
                                         sample_local_walkers=True)
    ctx = TableContext(jnp.arange(3), jnp.asarray(p), sampling)
    ctx = jax.tree.map(lambda x: replicate(np.asarray(x), mesh), ctx)
    inputs = (shard_first_axis(table, mesh), shard_first_axis(pi, mesh), shard_first_axis(q, mesh))
    zero = replicate(np.asarray(0.), mesh)

    def sample(key):
        return jnp.stack(local_pair_tail(mesh, *inputs, key, zero, ctx, zero,
                                        pair_fn=table_terms, n_chunks=2))

    keys = replicate(np.asarray(jax.random.split(jax.random.PRNGKey(93), 8192)), mesh)
    samples = np.asarray(jax.jit(jax.vmap(sample))(keys))
    exact, variance = 0., 0.
    for d in range(4):
        sl = slice(2*d, 2*d+2)
        if pi[sl].sum() == 0:
            continue
        qd = q[sl] / q[sl].sum()
        probabilities = qd[:, None] * p[None, :]
        values = pi[sl, None] * table[sl] / probabilities
        mean = np.sum(probabilities * values)
        exact += mean
        variance += (np.sum(probabilities * values**2) - mean**2) / (5 if d == 0 else 4)
    assert abs(samples[:, 0].mean() - exact) < 6 * np.sqrt(variance / len(samples))
    assert abs(samples[:, 1].mean()) < 6 * np.sqrt(variance / len(samples))
    np.testing.assert_allclose(samples.var(axis=0, ddof=1), variance, rtol=.07)


def molecular_inputs(kind, *, mixed=False, head=3):
    # Reuse the existing symmetry-correct trial fixtures, including active-space
    # offsets for CISD and unequal alpha/beta occupation counts for UCISD.
    if kind == "cisd":
        from tests.test_cisd_modes import _make_dense_and_mode_trials
        _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3, nocc_t_core=1,
            nvir_t_outer=1, mode_dtype=jnp.float32 if mixed else jnp.float64)
        module, cfg_type = cm, cm.CisdModePairSamplingCfg
        sys = System(trial.norb, (trial.nocc_full,)*2, "restricted")
        make_ops = cm.make_cisd_mode_meas_ops
    else:
        from tests.test_ucisd_k_modes import _make_trials
        _, trial, _ = _make_trials(norb=5, noa=2, nob=1, rank=5,
                                  mode_dtype=jnp.float32 if mixed else jnp.float64)
        module, cfg_type = um, um.UcisdKModePairSamplingCfg
        sys = System(trial.norb, trial.nocc, "restricted")
        make_ops = um.make_ucisd_k_mode_meas_ops
    rng = np.random.default_rng(418)
    n = trial.norb
    chol = .01 * rng.normal(size=(7, n, n))
    chol = (chol + chol.transpose(0, 2, 1)) / 2
    ham = HamChol(jnp.asarray(.2), jnp.diag(jnp.linspace(-1., 1., n)), jnp.asarray(chol))
    sampling = cfg_type(head, 17, head_chol_batch_size=2, guide_chol_batch_size=2,
        walker_guide_policy="head_rms", guard_head_deviations=True,
        track_half_sample_diagnostic=True, sample_local_walkers=True)
    ops = make_ops(sys, mixed_precision=mixed, n_mode_chunks=3, energy_sampling=sampling)
    ctx = ops.build_meas_ctx(ham, trial)
    walkers = np.eye(n, max(sys.nelec))[None] + .025 * (
        rng.normal(size=(12, n, max(sys.nelec))) + 1j*rng.normal(size=(12, n, max(sys.nelec))))
    weights = np.arange(1, 13, dtype=float)
    return module, ham, trial, ctx, walkers, weights


def molecular_reference(kind, ham, trial, ctx, walkers, weights, threshold):
    """Direct full-term sum with independently normalized shard proposals."""
    w = jnp.asarray(walkers)
    if kind == "cisd":
        common = jax.vmap(cm._cisd_mode_energy_common, in_axes=(0,None,None,None))(w, ham, ctx, trial)
        terms = cm._cisd_mode_chol_terms_for_walkers(
            common, ham.chol, ctx.rot_chol, ctx.lci1, ctx, trial)
    else:
        common = jax.vmap(lambda wi: um._ucisd_k_energy_common(
            wi, ham, ctx, trial, um._k_mode_apply_realimag))(w)
        terms = um._ucisd_k_mode_chol_terms_for_walkers(
            common, ham, ctx, trial, chol_indices=jnp.arange(7))
    terms = np.real(terms)
    head_terms = terms[:, np.asarray(ctx.chol_head_indices)]
    head = np.real(common.base) + head_terms.sum(axis=1)
    pi = weights / weights.sum()
    deviations = np.abs(head - np.sum(pi * head))
    if threshold is None:
        threshold = float(np.median(deviations))
    guarded = deviations > threshold
    target = np.where(guarded, 0., pi)
    accepted = target / target.sum()
    guide = target * np.sqrt(np.sum(head_terms**2, axis=1))
    guide /= guide.sum()
    mix = ctx.energy_sampling.walker_guide_weight_mix
    q = mix * accepted + (1 - mix) * guide
    energy = np.sum(pi * np.where(guarded, -2., head))
    noise = 0.
    p = np.asarray(ctx.chol_tail_prob)
    for d in range(4):
        sl = slice(3*d, 3*d+3)
        if target[sl].sum() == 0:
            continue
        qd = q[sl] / q[sl].sum()
        kw, kg = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(83), d))
        uniform_w = np.asarray(jax.random.uniform(kw, (5,), dtype=jnp.float64))
        uniform_g = np.asarray(jax.random.uniform(kg, (5,), dtype=jnp.float64))
        iw = np.searchsorted(np.cumsum(qd), uniform_w * qd.sum(), side="right")
        ig = np.searchsorted(np.cumsum(p), uniform_g * p.sum(), side="right")
        count = 5 if d == 0 else 4
        values = target[sl][iw] * terms[3*d+iw, np.asarray(ctx.chol_tail_indices)[ig]] / (qd[iw] * p[ig])
        values = values[:count]
        energy += values.mean()
        first = count // 2
        noise += np.sqrt(first * (count - first)) / count * (values[:first].mean() - values[first:].mean())
    return energy, noise, threshold, guarded.sum()


@pytest.mark.parametrize("kind", ["cisd", "ucisd"])
@pytest.mark.parametrize("mixed", [False, True])
def test_energy_head_guards_chunks_and_communication(kind, mixed):
    mesh = mesh4()
    module, ham, trial, ctx, walkers, weights = molecular_inputs(kind, mixed=mixed)
    w = shard_first_axis(walkers, mesh)
    weights_s = shard_first_axis(weights, mesh)
    h, c, t = jax.tree.map(lambda x: replicate(np.asarray(x), mesh), (ham, ctx, trial))
    overlap = shard_first_axis(np.ones(12, dtype=complex), mesh)
    key = replicate(np.asarray(jax.random.PRNGKey(83)), mesh)
    eref = replicate(np.asarray(-2.), mesh)
    threshold = replicate(np.asarray(100.), mesh)
    args = (w, weights_s, overlap, key, h, c, t, eref, threshold)
    outputs = []
    for chunks in (1, 2):
        fn = lambda w, wt, ov, k, h, c, t, e, th: module.pair_sampled_block_energy(
            w, wt, ov, k, chunks, h, c, t, e, th)
        compiled = jax.jit(fn).lower(*args).compile()
        assert_scalar_collectives(compiled)
        out = compiled(*args)
        assert np.isfinite(out.energy)
        outputs.append(out)
    tol = 1e-5 if mixed else 2e-11
    for a, b in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol)
    # Check the GLOBAL guard and RMS proposal diagnostics against the existing
    # single-device estimator, without requiring their random tail samples to match.
    legacy = replace(ctx, energy_sampling=replace(ctx.energy_sampling, sample_local_walkers=False))
    expected = jax.jit(module.pair_sampled_block_energy, static_argnums=4)(
        jnp.asarray(walkers), jnp.asarray(weights), jnp.ones(12, dtype=complex),
        jax.random.PRNGKey(83), 1, ham, legacy, trial, jnp.asarray(-2.), jnp.asarray(100.))
    for diagnostic in expected.diagnostics:
        if diagnostic != d_energy_sampling_noise:
            np.testing.assert_allclose(outputs[0].diagnostics[diagnostic], expected.diagnostics[diagnostic],
                                       rtol=tol, atol=tol)
    reference = molecular_reference(kind, ham, trial, ctx, walkers, weights, 100.)
    np.testing.assert_allclose([outputs[0].energy, outputs[0].diagnostics[d_energy_sampling_noise]],
                               reference[:2], rtol=tol, atol=tol)
    reference = molecular_reference(kind, ham, trial, ctx, walkers, weights, None)
    partial_guard = compiled(*args[:-1], replicate(np.asarray(reference[2]), mesh))
    np.testing.assert_allclose([partial_guard.energy, partial_guard.diagnostics[d_energy_sampling_noise]],
                               reference[:2], rtol=tol, atol=tol)
    assert 0 < reference[3] < 12
    assert partial_guard.diagnostics['energy_head_guard_count'] == reference[3]
    # Enabling the option on a single device preserves the original RNG path.
    serial_local = jax.jit(module.pair_sampled_block_energy, static_argnums=4)(
        jnp.asarray(walkers), jnp.asarray(weights), jnp.ones(12, dtype=complex),
        jax.random.PRNGKey(83), 1, ham, ctx, trial, jnp.asarray(-2.), jnp.asarray(100.))
    for a, b in zip(jax.tree.leaves(serial_local), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    # A negative threshold deliberately guards every walker. Tail and noise
    # must vanish, while the head replacement retains the global normalization.
    guarded = compiled(*args[:-1], replicate(np.asarray(-1.), mesh))
    np.testing.assert_allclose(guarded.energy, -2., atol=1e-12)
    assert float(guarded.diagnostics[d_energy_sampling_noise]) == 0.


@pytest.mark.parametrize("kind", ["cisd", "ucisd"])
def test_full_head_and_one_device_preserve_deterministic_result(kind):
    mesh = mesh4()
    module, ham, trial, ctx, walkers, weights = molecular_inputs(kind, head=7)
    h, c, t = jax.tree.map(lambda x: replicate(np.asarray(x), mesh), (ham, ctx, trial))
    fn = jax.jit(module.pair_sampled_block_energy, static_argnums=4)
    args = (jnp.asarray(walkers), jnp.asarray(weights), jnp.ones(12, dtype=complex),
            jax.random.PRNGKey(83), 2, ham, ctx, trial, jnp.asarray(-2.), jnp.asarray(100.))
    single = fn(*args)
    distributed = fn(shard_first_axis(walkers, mesh), shard_first_axis(weights, mesh),
        shard_first_axis(np.ones(12, dtype=complex), mesh), replicate(np.asarray(args[3]), mesh),
        2, h, c, t, replicate(np.asarray(-2.), mesh), replicate(np.asarray(100.), mesh))
    exact = jax.vmap(module.energy_kernel_rw_rh, in_axes=(0,None,None,None))(
        jnp.asarray(walkers), ham, ctx, trial)
    expected = np.sum(weights * np.real(exact)) / weights.sum()
    np.testing.assert_allclose([single.energy, distributed.energy], expected, rtol=2e-11, atol=2e-11)
    assert float(distributed.diagnostics[d_energy_sampling_noise]) == 0.


def test_model_sharding_is_not_silently_replicated():
    devices = np.asarray(mesh4().devices).reshape(2, 2)
    mesh = Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))
    walkers = shard_first_axis(np.zeros((8, 3, 2)), mesh)
    with pytest.raises(ValueError, match="replicated Hamiltonian"):
        local_pair_mesh(walkers, True)


@pytest.mark.parametrize("kind", ["cisd", "ucisd"])
def test_weight_only_sampler_without_diagnostics_and_small_budget(kind):
    mesh = mesh4()
    module, ham, trial, ctx, walkers, weights = molecular_inputs(kind, head=0)
    sampling = replace(ctx.energy_sampling, walker_guide_policy="weight",
                       track_half_sample_diagnostic=False, guard_head_deviations=False,
                       pair_sample_size=4)
    ctx = replace(ctx, energy_sampling=sampling)
    h, c, t = jax.tree.map(lambda x: replicate(np.asarray(x), mesh), (ham, ctx, trial))
    args = (shard_first_axis(walkers, mesh), shard_first_axis(weights, mesh),
            shard_first_axis(np.ones(12, dtype=complex), mesh),
            replicate(np.asarray(jax.random.PRNGKey(83)), mesh), 1, h, c, t,
            replicate(np.asarray(-2.), mesh), replicate(np.asarray(100.), mesh))
    fn = jax.jit(module.pair_sampled_block_energy, static_argnums=4)
    result = fn(*args)
    assert np.shape(result) == () and np.isfinite(result)
    bad = replace(c, energy_sampling=replace(sampling, pair_sample_size=3))
    with pytest.raises(ValueError, match="at least 1 pairs per data shard"):
        fn(*args[:6], bad, *args[7:])
    bad = replace(c, energy_sampling=replace(sampling, track_half_sample_diagnostic=True))
    with pytest.raises(ValueError, match="at least 2 pairs per data shard"):
        fn(*args[:6], bad, *args[7:])


@pytest.mark.parametrize("kind", ["cisd", "ucisd"])
def test_local_sampling_rejects_global_variance_retuning(kind):
    module, ham, trial, ctx, _, _ = molecular_inputs(kind)
    if kind == "cisd":
        sys = System(trial.norb, (trial.nocc_full,)*2, "restricted")
        make = module.make_cisd_mode_meas_ops
        tuning = module.CisdModePairTuningCfg()
    else:
        sys = System(trial.norb, trial.nocc, "restricted")
        make = module.make_ucisd_k_mode_meas_ops
        tuning = module.UcisdKModePairTuningCfg()
    with pytest.raises(ValueError, match="frozen settings"):
        make(sys, energy_sampling=ctx.energy_sampling, energy_tuning=tuning)
    retune = (cm.retune_cisd_mode_pair_sampling if kind == "cisd"
              else um.retune_ucisd_k_mode_pair_sampling)
    with pytest.raises(ValueError, match="frozen settings"):
        retune(None, None, None, None, ham, ctx, trial,
               advance_blocks=None, tuning_cfg=tuning)
