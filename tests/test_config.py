"""Check global precision in fresh processes without running AFQMC."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "precision,environment,import_jax_first,expected",
    [
        (None, None, False, "highest"),
        (None, None, True, "highest"),
        ("default", None, False, "default"),
        ("high", None, False, "high"),
        (None, "default", False, "default"),
        ("default", "highest", False, "default"),
        ("highest", "default", True, "highest"),
    ],
)
def test_matmul_precision_configuration(precision, environment, import_jax_first, expected):
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
    env.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
    if environment is not None:
        env["JAX_DEFAULT_MATMUL_PRECISION"] = environment
    code = f"""
from trot import config
if {import_jax_first!r}:
    import jax
config.configure_once(use_gpu=False, matmul_precision={precision!r})
import jax
import jax.numpy as jnp
assert jax.config.jax_default_matmul_precision == {expected!r}
assert config.afqmc_config.matmul_precision == {expected!r}

# Inspect a production-style contraction's tracing metadata, without
# compiling or executing numerical work on the CPU.
operand = jax.ShapeDtypeStruct((3, 3), jnp.float32)
traced = jax.make_jaxpr(lambda a, b: jnp.einsum('ik,kj->ij', a, b))(operand, operand)
dot, = [eq for eq in traced.jaxpr.eqns if eq.primitive.name == 'dot_general']
assert all(p.name.lower() == {expected!r} for p in dot.params['precision'])
assert traced.out_avals[0].dtype == jnp.dtype('float32')
assert jax.config.jax_enable_x64

# Import-time setup in other modules must preserve the first configuration.
config.configure_once(matmul_precision={'default' if expected == 'highest' else 'highest'!r})
assert jax.config.jax_default_matmul_precision == {expected!r}
assert config.afqmc_config.matmul_precision == {expected!r}
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
