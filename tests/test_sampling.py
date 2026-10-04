"""Finite-law tests include kernel checks of true and deliberately false claims."""
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxlean import Discrete, TranslationError
from examples.sampling import stages

ROOT = Path(__file__).resolve().parents[1]


def check_lean(source, filename):
    path = ROOT / "tests/_generated" / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    return subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


def test_all_stages_and_exact_probabilities():
    for _, (_, rv, mean, variance) in stages().items():
        values, probabilities = rv.enumerate()
        assert sum(probabilities) == 1
        assert all(p >= 0 for p in probabilities)
        # These examples deliberately use small integers, exact in float32.
        exact_values = [Fraction(float(v)) for v in values]
        exact_mean = sum(p * v for p, v in zip(probabilities, exact_values))
        exact_variance = sum(p * (v - exact_mean) ** 2 for p, v in zip(probabilities, exact_values))
        assert exact_mean == mean
        assert exact_variance == variance
        np.testing.assert_allclose(rv.moments(), [float(mean), float(variance)])
    average = stages()["AverageDraw"][1]
    values, probabilities = average.enumerate()
    assert values.tolist() == [1, 13, 13, 25]
    assert probabilities == (Fraction(1, 16), Fraction(3, 16), Fraction(3, 16), Fraction(9, 16))


def test_shared_and_independent_draws_differ():
    x = Discrete([0, 2])
    independent = x.independent_map2(x, lambda a, b: a + b)
    reused = x.map(lambda v: v + v)
    assert tuple(map(float, independent.moments())) == (2, 2)
    assert tuple(map(float, reused.moments())) == (2, 4)


def test_jit_sampling_shapes_keys_and_zero_mass():
    rv = stages()["AverageDraw"][1]
    fn = jax.jit(lambda key: rv.sample(key, (4, 5)))
    key = jax.random.key(42)
    a, b = fn(key), fn(key)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 5)
    assert set(np.asarray(a).flat) <= {1, 13, 25}
    assert rv.sample(key).shape == ()
    assert rv.sample(key, (0, 2)).shape == (0, 2)
    certain = Discrete([-100, 7], weights=[0, 5]).map(lambda v: v + 1)
    np.testing.assert_array_equal(certain.sample(key, (100,)), np.full(100, 8))


@pytest.mark.parametrize("values,weights", [
    ([], None), ([[1, 2]], None), ([True, False], None), ([1j], None),
    ([float("nan")], None), ([float("inf")], None), ([1e100], None),
    ([1, 2], [1]), ([1, 2], [0, 0]), ([1, 2], [-1, 2]),
    ([1, 2], [0.5, 0.5]), ([1, 2], [True, False]),
])
def test_invalid_laws(values, weights):
    with pytest.raises(ValueError):
        Discrete(values, weights)


def test_reject_unsupported_operations_and_bound_enumeration():
    x = Discrete([0, 1])
    with pytest.raises(ValueError, match="floating scalar"):
        x.map(lambda v: jnp.array([v, v]))
    with pytest.raises(ValueError, match="floating scalar"):
        x.map(lambda v: jnp.int32(1))
    with pytest.raises(TranslationError):
        x.map(lambda v: jnp.floor(v))
    with pytest.raises(TranslationError):
        x.map(lambda v: v + jax.random.normal(jax.random.key(0)))
    with pytest.raises(ValueError, match="limit"):
        x.independent_map2(x, lambda a, b: a + b).enumerate(max_outcomes=3)
    with pytest.raises(ValueError, match="exact claims"):
        x.to_lean(mean=0.5)
    with pytest.raises(TranslationError, match="identifier"):
        x.to_lean(name="bad;name")


@pytest.mark.lean
def test_lean_checks_all_generated_moments():
    subprocess.run([sys.executable, "-m", "examples.sampling", "--check"], cwd=ROOT, check=True)
    modules = "\n".join(f"import JaxLean.Generated.{module}" for module in stages())
    # Build, rather than only importing potentially stale compiled modules.
    result = subprocess.run(["lake", "build", *[f"JaxLean.Generated.{m}" for m in stages()]],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = check_lean(modules, "SamplingImports.lean")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_lean_rejects_false_moment_claim():
    source = Discrete([0, 2]).to_lean(name="falseClaim", mean=2)
    result = check_lean(source, "FalseMoment.lean")
    assert result.returncode != 0
    assert "unsolved goals" in result.stdout


@pytest.mark.lean
def test_export_preserves_arithmetic_instead_of_precomputing_floats():
    rv = Discrete([2**24]).map(lambda v: (v + 1) - v)
    assert float(rv.moments()[0]) == 0  # Float32 rounded away the increment.
    source = rv.to_lean(name="realArithmetic", mean=1, variance=0)
    result = check_lean(source, "RealSamplingArithmetic.lean")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_zero_weights_duplicate_values_and_nonuniform_lean():
    rv = Discrete([1, 1, 4], weights=[0, 2, 1]).map(lambda v: v * v)
    result = check_lean(rv.to_lean(name="duplicates", namespace="Proof.end", mean=6, variance=50),
                        "DuplicateSampling.lean")
    assert result.returncode == 0, result.stdout + result.stderr
