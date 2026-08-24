from __future__ import annotations

from dataclasses import replace

from trot import config

config.configure_once(use_gpu=False)

import h5py
import jax
import numpy as np
import pytest

from trot.afqmc import Afqmc
from trot.cisd_workflow import (
    CisdModeConfig,
    CisdPairSamplingConfig,
    CisdWorkflowConfig,
    prepare_cisd_modes,
)
from trot.ham.chol import HamChol
from trot.meas.cisd_modes import get_cisd_mode_meas_cfg
from trot.meas.ucisd_k_modes import get_ucisd_k_mode_meas_cfg
from trot.setup import setup
from trot.staging import HamInput, StagedInputs, TrialInput, dump, load
from trot.trial.cisd import CisdTrial
from trot.trial.cisd_modes import CisdModeTrial
from trot.trial.ucisd_k_modes import UcisdKModeTrial


def _ham_input(*, norb: int, nelec: tuple[int, int], seed: int = 91) -> HamInput:
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = rng.standard_normal((5, norb, norb))
    chol = 0.5 * (chol + chol.transpose(0, 2, 1))
    return HamInput(
        h0=0.25,
        h1=h1,
        chol=chol,
        nelec=nelec,
        norb=norb,
        chol_cut=1.0e-5,
        frozen=0,
        source_kind="cc",
        basis="restricted",
    )


def _staged_cisd(seed: int = 103) -> StagedInputs:
    rng = np.random.default_rng(seed)
    norb = 4
    nocc = 2
    nvir = norb - nocc
    ci1 = 0.03 * rng.standard_normal((nocc, nvir))
    ci2_pair = 0.02 * rng.standard_normal((nocc * nvir, nocc * nvir))
    ci2_pair = 0.5 * (ci2_pair + ci2_pair.T)
    ci2 = ci2_pair.reshape(nocc, nvir, nocc, nvir)
    trial = TrialInput(
        kind="cisd",
        data={
            "ci1": ci1,
            "ci2": ci2,
            "nocc_t_core": np.asarray(0, dtype=np.int64),
            "nvir_t_outer": np.asarray(0, dtype=np.int64),
        },
        frozen=0,
        source_kind="cc",
    )
    return StagedInputs(
        ham=_ham_input(norb=norb, nelec=(nocc, nocc), seed=seed + 1),
        trial=trial,
        meta={
            "format_version": 1,
            "source_kind": "cc",
            "frozen": 0,
            "chol_cut": 1.0e-5,
        },
    )


def _staged_ucisd(seed: int = 127) -> StagedInputs:
    rng = np.random.default_rng(seed)
    norb = 4
    noa, nob = 2, 1
    nva, nvb = norb - noa, norb - nob
    trial = TrialInput(
        kind="ucisd",
        data={
            "mo_coeff_a": np.eye(norb),
            "mo_coeff_b": np.eye(norb),
            "ci1a": 0.03 * rng.standard_normal((noa, nva)),
            "ci1b": 0.03 * rng.standard_normal((nob, nvb)),
            "ci2aa": np.zeros((noa, nva, noa, nva)),
            "ci2ab": 0.02 * rng.standard_normal((noa, nva, nob, nvb)),
            "ci2bb": np.zeros((nob, nvb, nob, nvb)),
        },
        frozen=0,
        source_kind="cc",
    )
    return StagedInputs(
        ham=_ham_input(norb=norb, nelec=(noa, nob), seed=seed + 1),
        trial=trial,
        meta={
            "format_version": 1,
            "source_kind": "cc",
            "frozen": 0,
            "chol_cut": 1.0e-5,
        },
    )


def test_prepare_cisd_modes_preserves_raw_and_round_trips_derived(tmp_path):
    path = tmp_path / "afqmc.h5"
    staged = _staged_cisd()
    dump(staged, path)
    mode_config = CisdModeConfig(
        threshold=None,
        discarded_norm_target=0.1,
        solver="dense",
    )

    prepared = prepare_cisd_modes(path, mode_config, verbose=False)

    raw = load(path)
    derived = load(path, derived_trial_key=prepared.cache_key)
    assert "ci2" in raw.trial.data
    assert "ci2" not in derived.trial.data
    assert {"ci1", "eigenvalues", "modes"} <= set(derived.trial.data)
    assert derived.meta["trial_representation"]["mode_rank"] == prepared.metadata["mode_rank"]
    with h5py.File(path, "r") as handle:
        assert f"trial/derived/{prepared.cache_key}/data/modes" in handle
        assert "trial/data/ci2" in handle


def test_prepare_ucisd_modes_preserves_raw_and_round_trips_derived(tmp_path):
    path = tmp_path / "afqmc.h5"
    staged = _staged_ucisd()
    dump(staged, path)
    mode_config = CisdModeConfig(
        threshold=None,
        discarded_norm_target=0.1,
        solver="dense",
    )

    prepared = prepare_cisd_modes(path, mode_config, verbose=False)
    raw = load(path)
    derived = load(path, derived_trial_key=prepared.cache_key)

    assert "ci2ab" in raw.trial.data
    assert "ci2ab" not in derived.trial.data
    assert {
        "mo_coeff_a",
        "mo_coeff_b",
        "c1a",
        "c1b",
        "eigenvalues",
        "modes",
    } <= set(derived.trial.data)


def test_multiple_mode_selections_coexist_in_one_stage_cache(tmp_path):
    path = tmp_path / "afqmc.h5"
    dump(_staged_cisd(), path)
    loose = CisdModeConfig(
        threshold=None,
        discarded_norm_target=0.2,
        solver="dense",
    )
    tight = replace(loose, discarded_norm_target=0.05)

    loose_modes = prepare_cisd_modes(path, loose, verbose=False)
    tight_modes = prepare_cisd_modes(path, tight, verbose=False)

    assert loose_modes.cache_key != tight_modes.cache_key
    with h5py.File(path, "r") as handle:
        derived = handle["trial/derived"]
        assert loose_modes.cache_key in derived
        assert tight_modes.cache_key in derived
        assert "ci2" in handle["trial/data"]


def test_setup_full_rank_cisd_modes_matches_dense_overlap():
    staged = _staged_cisd()
    dense = setup(
        staged,
        walker_kind="restricted",
        mixed_precision=False,
    )
    workflow = CisdWorkflowConfig(
        modes=CisdModeConfig(
            threshold=None,
            discarded_norm_target=0.0,
            solver="dense",
        )
    )
    mode = setup(
        staged,
        walker_kind="restricted",
        mixed_precision=False,
        cisd_workflow=workflow,
    )

    assert isinstance(dense.trial_data, CisdTrial)
    assert isinstance(mode.trial_data, CisdModeTrial)
    walker = np.eye(staged.ham.norb, staged.ham.nelec[0], dtype=np.complex128)
    walker[2:, :] += np.asarray([[0.12j, 0.03], [-0.04, 0.09j]])
    dense_overlap = dense.trial_ops.overlap(jax.numpy.asarray(walker), dense.trial_data)
    mode_overlap = mode.trial_ops.overlap(jax.numpy.asarray(walker), mode.trial_data)
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=1.0e-11, atol=1.0e-11)
    assert get_cisd_mode_meas_cfg(mode.meas_ops) is not None
    assert mode.meas_ops.block_energy is None


@pytest.mark.parametrize(
    "staged_factory,trial_type,cfg_getter",
    [
        (_staged_cisd, CisdModeTrial, get_cisd_mode_meas_cfg),
        (_staged_ucisd, UcisdKModeTrial, get_ucisd_k_mode_meas_cfg),
    ],
)
def test_setup_pair_sampled_cisd_workflow(
    staged_factory,
    trial_type,
    cfg_getter,
):
    workflow = CisdWorkflowConfig(
        modes=CisdModeConfig(
            threshold=None,
            discarded_norm_target=0.1,
            solver="dense",
        ),
        pair_sampling=CisdPairSamplingConfig(
            initial_pair_sample_size=32,
            tuning_population_count=2,
        ),
    )
    job = setup(
        staged_factory(),
        walker_kind="restricted",
        mixed_precision=False,
        cisd_workflow=workflow,
    )

    assert isinstance(job.trial_data, trial_type)
    assert cfg_getter(job.meas_ops) is not None
    assert job.meas_ops.block_energy is not None
    assert job.meas_ops.retune_block_energy is not None
    assert job.staged.meta["cisd_workflow"]["pair_sampling"] is not None
    state, meas_ctx, prop_ctx = job._prepare_runtime()
    assert state.walkers.shape[0] == job.params.n_walkers
    assert meas_ctx.energy_sampling is not None
    assert prop_ctx is not None


def test_afqmc_from_staged_selects_cached_modes_without_raw_doubles(tmp_path):
    path = tmp_path / "afqmc.h5"
    staged = _staged_cisd()
    dump(staged, path)
    workflow = CisdWorkflowConfig(
        modes=CisdModeConfig(
            threshold=None,
            discarded_norm_target=0.1,
            solver="dense",
        )
    )
    prepared = prepare_cisd_modes(path, workflow.modes, verbose=False)

    af = Afqmc.from_staged(path, cisd_workflow=workflow)
    assert af.staged is not None
    assert "ci2" not in af.staged.trial.data
    assert af.staged.meta["trial_representation"]["cache_key"] == prepared.cache_key
    af.walker_kind = "restricted"
    af.mixed_precision = False
    job = af.build_job()
    assert isinstance(job.trial_data, CisdModeTrial)


def test_afqmc_prepares_modes_alongside_raw_trial(tmp_path):
    path = tmp_path / "afqmc.h5"
    dump(_staged_ucisd(), path)
    workflow = CisdWorkflowConfig(
        modes=CisdModeConfig(
            threshold=None,
            discarded_norm_target=0.1,
            solver="dense",
        )
    )
    af = Afqmc.from_staged(path, cisd_workflow=workflow)

    prepared = af.prepare_cisd_trial_cache()

    raw = load(path)
    derived = load(path, derived_trial_key=prepared.cache_key)
    assert "ci2ab" in raw.trial.data
    assert "ci2ab" not in derived.trial.data
    assert prepared.metadata["discarded_norm_target"] == pytest.approx(0.1)


def test_save_staged_preserves_raw_and_cached_modes(tmp_path):
    source = tmp_path / "afqmc.h5"
    destination = tmp_path / "copy.h5"
    dump(_staged_cisd(), source)
    workflow = CisdWorkflowConfig(
        modes=CisdModeConfig(
            threshold=None,
            discarded_norm_target=0.1,
            solver="dense",
        )
    )
    af = Afqmc.from_staged(source, cisd_workflow=workflow)
    prepared = af.prepare_cisd_trial_cache()

    af.save_staged(destination)

    assert "ci2" in load(destination).trial.data
    assert "ci2" not in load(
        destination,
        derived_trial_key=prepared.cache_key,
    ).trial.data


def test_cisd_workflow_rejects_non_cisd_trial():
    staged = _staged_cisd()
    staged = replace(staged, trial=replace(staged.trial, kind="rhf"))
    with pytest.raises(ValueError, match="requires staged trial kind"):
        setup(
            staged,
            walker_kind="restricted",
            cisd_workflow=CisdWorkflowConfig(),
        )


def test_cisd_workflow_rejects_manual_trial_overrides():
    with pytest.raises(ValueError, match="cannot be combined"):
        setup(
            _staged_cisd(),
            walker_kind="restricted",
            cisd_workflow=CisdWorkflowConfig(),
            trial_data=object(),
        )
