import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from trot.ham.chol import HamChol, HamCholUhf
from trot.prop.chol_afqmc_ops import UhfCholAfqmcCtx
from trot.prop.types import QmcParams
from trot.runtime_layout import DefaultRuntimeLayout
from trot.setup import setup
from trot.staging import HamInput, StagedInputs, TrialInput, dump, load


def test_regular_ham_input_roundtrip_keeps_legacy_hdf5_layout(tmp_path):
    norb, nchol = 3, 4
    ham = HamInput(
        h0=-0.5,
        h1=np.eye(norb),
        chol=np.zeros((nchol, norb, norb)),
        nelec=(1, 1),
        norb=norb,
        chol_cut=1.0e-5,
        frozen=0,
        source_kind="cc",
        basis="restricted",
    )
    trial = TrialInput(
        kind="uhf",
        data={
            "mo_a": np.eye(norb)[:, :1],
            "mo_b": np.eye(norb)[:, :1],
        },
        frozen=0,
        source_kind="cc",
    )
    path = tmp_path / "legacy_restricted.h5"
    dump(StagedInputs(ham=ham, trial=trial, meta={"format_version": 1}), path)

    with h5py.File(path, "r") as handle:
        assert "h1_b" not in handle["ham"]
        assert "chol_b" not in handle["ham"]
        assert "norb_spin" not in handle["ham"]

    loaded = load(path)
    assert loaded.ham.h1_b is None
    assert loaded.ham.chol_b is None
    assert loaded.ham.norb_spin is None
    runtime = DefaultRuntimeLayout().make_initial_ham_data(loaded.ham, mesh=None)
    assert isinstance(runtime, HamChol)


def test_split_uhf_ham_input_roundtrip_and_runtime_conversion(tmp_path):
    rng = np.random.default_rng(73)
    norb, nchol = 4, 6
    h1_a = rng.standard_normal((norb, norb))
    h1_a = 0.5 * (h1_a + h1_a.T)
    h1_b = rng.standard_normal((norb, norb))
    h1_b = 0.5 * (h1_b + h1_b.T)
    h1_b[-1] = 0.0
    h1_b[:, -1] = 0.0
    chol_a = rng.standard_normal((nchol, norb, norb))
    chol_a = 0.5 * (chol_a + chol_a.transpose(0, 2, 1))
    chol_b = rng.standard_normal((nchol, norb, norb))
    chol_b = 0.5 * (chol_b + chol_b.transpose(0, 2, 1))
    chol_b[:, -1] = 0.0
    chol_b[:, :, -1] = 0.0

    ham = HamInput(
        h0=-1.25,
        h1=h1_a,
        chol=chol_a,
        nelec=(2, 1),
        norb=norb,
        chol_cut=1.0e-5,
        frozen=0,
        source_kind="cc",
        basis="unrestricted",
        h1_b=h1_b,
        chol_b=chol_b,
        norb_spin=(4, 3),
    )
    trial = TrialInput(
        kind="uhf",
        data={
            "mo_a": np.eye(norb)[:, :2],
            "mo_b": np.eye(norb)[:, :1],
        },
        frozen=0,
        source_kind="cc",
    )
    staged = StagedInputs(ham=ham, trial=trial, meta={"format_version": 1})
    path = tmp_path / "split_uhf.h5"
    dump(staged, path)
    loaded = load(path)

    assert loaded.ham.basis == "unrestricted"
    assert loaded.ham.norb_spin == (4, 3)
    np.testing.assert_array_equal(loaded.ham.h1, h1_a)
    np.testing.assert_array_equal(loaded.ham.h1_b, h1_b)
    np.testing.assert_array_equal(loaded.ham.chol, chol_a)
    np.testing.assert_array_equal(loaded.ham.chol_b, chol_b)

    runtime = DefaultRuntimeLayout().make_initial_ham_data(loaded.ham, mesh=None)
    assert isinstance(runtime, HamCholUhf)
    assert runtime.nchol == nchol
    assert runtime.norb_spin == (4, 3)
    np.testing.assert_array_equal(runtime.h1_a, h1_a)
    np.testing.assert_array_equal(runtime.h1_b, h1_b)
    np.testing.assert_array_equal(runtime.chol_a, chol_a)
    np.testing.assert_array_equal(runtime.chol_b, chol_b)

    job = setup(
        loaded,
        mixed_precision=False,
        params=QmcParams(
            n_walkers=2,
            n_eql_blocks=0,
            n_blocks=1,
            n_prop_steps=1,
            seed=79,
        ),
    )
    state, meas_ctx, prop_ctx = job._prepare_runtime()
    assert state.walkers[0].shape == (2, norb, 2)
    assert state.walkers[1].shape == (2, norb, 1)
    assert meas_ctx.rot_chol_a.shape == (nchol, 2, norb)
    assert meas_ctx.rot_chol_b.shape == (nchol, 1, norb)
    assert isinstance(prop_ctx, UhfCholAfqmcCtx)
    assert isinstance(job.ham_data, HamCholUhf)
    assert job.ham_data.chol_a.shape == (0, 0, 0)
    assert job.ham_data.chol_b.shape == (0, 0, 0)
