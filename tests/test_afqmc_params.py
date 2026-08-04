import pytest

from trot.afqmc import Afqmc, AfqmcFp
from trot.prop.types import QmcParams, QmcParamsFp


class _DummyMF:
    pass


DUMMY_MF = _DummyMF()


def test_afqmc_defaults_match_qmc_params():
    af = Afqmc(DUMMY_MF)
    defaults = QmcParams()
    assert af.dt == defaults.dt
    assert af.n_walkers == defaults.n_walkers
    assert af.n_blocks == defaults.n_blocks
    assert af.n_eql_blocks == defaults.n_eql_blocks
    assert isinstance(af.seed, int)
    assert af.n_chunks == defaults.n_chunks
    assert af.error_method == defaults.error_method == "gamma"


def test_afqmc_custom_values_override_defaults():
    af = Afqmc(
        DUMMY_MF,
        dt=0.01,
        n_walkers=50,
        n_blocks=100,
        seed=786,
        n_chunks=4,
        error_method="blocking",
    )
    assert af.dt == 0.01
    assert af.n_walkers == 50
    assert af.n_blocks == 100
    assert af.seed == 786
    assert af.n_chunks == 4
    assert af.error_method == "blocking"


def test_afqmc_partial_overrides():
    af = Afqmc(DUMMY_MF, dt=0.02)
    defaults = QmcParams()
    assert af.dt == 0.02
    assert af.n_walkers == defaults.n_walkers
    assert isinstance(af.seed, int)


def test_afqmc_params_starts_none():
    af = Afqmc(DUMMY_MF)
    assert af.params is None


def test_afqmc_source_kind_mf():
    af = Afqmc(DUMMY_MF)
    assert af.source_kind == "mf"


def test_afqmc_make_params_from_self_attributes():
    af = Afqmc(DUMMY_MF, dt=0.01, n_walkers=50, seed=7)
    params = af._make_params()
    assert isinstance(params, QmcParams)
    assert params.dt == 0.01
    assert params.n_walkers == 50
    assert params.seed == 7


def test_afqmc_make_params_uses_defaults():
    af = Afqmc(DUMMY_MF)
    params = af._make_params()
    defaults = QmcParams(seed=params.seed)
    assert params == defaults


def test_afqmc_make_params_respects_user_provided():
    custom = QmcParams(dt=0.1, n_walkers=10, seed=123)
    af = Afqmc(DUMMY_MF)
    af.params = custom
    params = af._make_params()
    assert params is custom


def test_afqmc_make_params_rejects_wrong_type():
    af = Afqmc(DUMMY_MF)
    af.params = "not a params"  # type: ignore
    with pytest.raises(TypeError, match="Expected type QmcParams"):
        af._make_params()


def test_afqmc_make_params_rejects_invalid_error_method():
    af = Afqmc(DUMMY_MF)
    af.error_method = "unknown"
    with pytest.raises(ValueError, match="error_method"):
        af._make_params()


def test_afqmcfp_defaults_match_qmc_params_fp():
    af = AfqmcFp(DUMMY_MF)
    defaults = QmcParamsFp()
    assert af.dt == defaults.dt
    assert af.n_walkers == defaults.n_walkers
    assert af.n_blocks == defaults.n_blocks
    assert af.n_prop_steps == defaults.n_prop_steps
    assert isinstance(af.seed, int)
    assert af.n_chunks == defaults.n_chunks
    assert af.n_traj == defaults.n_traj
    assert af.ene0 is None


def test_afqmcfp_make_params_requires_ene0():
    af = AfqmcFp(DUMMY_MF)
    with pytest.raises(ValueError, match="ene0"):
        af._make_params()
