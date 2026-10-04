"""Ordinary JAX -> core translator -> stdlib proof, without a distribution DSL."""
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from jaxlean import translate, transpile, TranslationError
from examples.random_program import sample_times_100, sample_scaled

ROOT = Path(__file__).resolve().parents[1]


def lean(source, filename):
    path = ROOT / "tests/_generated" / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    return subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


def test_source_is_standard_jax_and_readable_output():
    # Executable unchanged, including jit; no jaxlean objects in the source.
    key = jax.random.key(0)
    value = jax.jit(sample_times_100)(key)
    assert float(value) in (0, 100, 200, 300, 400, 500)
    source = transpile(sample_scaled, key, jnp.float32(100), random_model="uniform")
    assert "«sample_scaled»" in source
    assert "(«scale» : R)" in source
    assert "random_program.py:" in source
    assert "SAMPLER SPECIFICATION" in source
    assert "Tensor" not in source
    assert "(«key»" not in source  # The model replaces a key by a law, explicitly.
    subprocess.run([sys.executable, "-m", "examples.transpile_random", "--check"], cwd=ROOT, check=True)


def test_stdlib_proof_checks_and_wrong_scaling_fails():
    good = """import JaxLean.RandomProgramProofs
open JaxLean JaxLean.Generated
example : (sample_times_100 (R := ℝ)).variance =
    100 ^ 2 * (Rand.uniformInt (R := ℝ) 0 6 (by decide)).variance := by
  simp only [sample_times_100, Rand.variance_map_mul]
"""
    result = lean(good, "RandomScaling.lean")
    assert result.returncode == 0, result.stdout + result.stderr
    bad = """import JaxLean.RandomProgramProofs
open JaxLean JaxLean.Generated
example : (sample_times_100 (R := ℝ)).variance = 100 := by
  rw [RandomProgramProofs.sample_times_100_variance_value]
  norm_num
"""
    result = lean(bad, "WrongRandomScaling.lean")
    assert result.returncode != 0
    assert "unsolved goals" in result.stdout


def test_new_source_different_bounds_and_scale_same_rule():
    def changed(key):
        draw = jax.random.randint(key, (), -3, 4)
        return draw.astype(jnp.float32) * 7.0

    # Explicit Jaxpr entrypoint and an outer jit both use the same core.
    closed = jax.make_jaxpr(jax.jit(changed))(jax.random.key(1))
    source = translate(closed, name="changed", random_model="uniform", readable=True)
    source += """
example : (Generated.changed (R := ℝ)).variance =
    7 ^ 2 * (JaxLean.Rand.uniformInt (R := ℝ) (-3) 7 (by decide)).variance := by
  simp only [Generated.changed, JaxLean.Rand.variance_map_mul]
"""
    result = lean(source, "ChangedRandomSource.lean")
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_is_opt_in():
    with pytest.raises(TranslationError):
        transpile(sample_times_100, jax.random.key(0))
    with pytest.raises(TranslationError, match="random_model"):
        transpile(sample_times_100, jax.random.key(0), random_model="unverified-magic")


def test_rejects_integer_arithmetic_and_multiple_draws():
    def integer_arithmetic(key):
        return jax.random.randint(key, (), 0, 6) * 100

    def twice(key):
        a = jax.random.randint(key, (), 0, 6).astype(jnp.float32)
        b = jax.random.randint(key, (), 0, 6).astype(jnp.float32)
        return a + b

    with pytest.raises(TranslationError, match="integer"):
        transpile(integer_arithmetic, jax.random.key(0), random_model="uniform")
    with pytest.raises(TranslationError, match="one draw"):
        transpile(twice, jax.random.key(0), random_model="uniform")


def test_rejects_counterfeit_randint_name():
    @jax.jit
    def _randint(key, lo, hi):
        return jax.random.randint(key, (), lo, hi) + 1

    def fake(key):
        return _randint(key, 0, 6).astype(jnp.float32) * 100.0

    with pytest.raises(TranslationError, match="does not match"):
        transpile(fake, jax.random.key(0), random_model="uniform")


@pytest.mark.parametrize("lo,hi", [(0, 0), (4, 2)])
def test_rejects_empty_uniform_spec(lo, hi):
    def source(key):
        return jax.random.randint(key, (), lo, hi).astype(jnp.float32)
    with pytest.raises(TranslationError, match="nonempty"):
        transpile(source, jax.random.key(0), random_model="uniform")


def test_readable_deterministic_scalar_and_tensor_paths():
    def affine(value, scale):
        return value * scale + 1.0
    scalar = transpile(affine, jnp.float32(2), jnp.float32(3), namespace="Scalar")
    scalar += '\nexample : Scalar.affine (R := ℚ) 2 3 = 7 := by norm_num [Scalar.affine]\n'
    # norm_num comes from the public proof library, not the generated program.
    scalar = "import JaxLean.Stdlib\n" + scalar
    result = lean(scalar, "ReadableScalar.lean")
    assert result.returncode == 0, result.stdout + result.stderr
    tensor = transpile(affine, jnp.ones(3), jnp.float32(3), namespace="Array")
    assert "«value» : Tensor R [3]" in tensor
    result = lean(tensor, "ReadableTensor.lean")
    assert result.returncode == 0, result.stdout + result.stderr


def test_scalar_source_with_array_intermediate_uses_tensor_path():
    def temporary_array(value):
        return jnp.sum(jnp.stack([value, value + 1.0]))
    source = transpile(temporary_array, jnp.float32(2))
    assert "Tensor R []" in source
    result = lean(source, "ScalarWithArrayIntermediate.lean")
    assert result.returncode == 0, result.stdout + result.stderr


def test_scalar_operations_and_name_collisions():
    def operations(arg1, R):
        # R is reserved for the Lean type; its fallback must not capture arg1.
        v = jnp.maximum(-arg1, R ** 2) / 2.0
        return jnp.where(v > 1.0, jnp.abs(v - 1.0), jnp.minimum(v, R))
    source = "import JaxLean.Stdlib\n" + transpile(
        operations, jnp.float32(-4), jnp.float32(3), namespace="ScalarOps")
    expected = float(operations(jnp.float32(-4), jnp.float32(3)))
    assert expected == 3.5
    source += """
example : ScalarOps.operations (R := ℚ) (-4) 3 = 7 / 2 := by
  norm_num [ScalarOps.operations]
"""
    result = lean(source, "ScalarOperations.lean")
    assert result.returncode == 0, result.stdout + result.stderr
