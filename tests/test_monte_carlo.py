"""Product-law abstraction and reusable proofs about actual transpiled Jaxpr."""
import jax
import jax.numpy as jnp
import pytest

from jaxlean import translate, transpile, TranslationError
from examples.monte_carlo.code import monte_carlo, monte_carlo_4, uniform_six
from test_random_translation import lean


@pytest.mark.lean
def test_standard_jax_and_generated_proofs():
    value = jax.jit(monte_carlo_4)(jax.random.key(0))
    assert 0 <= float(value) <= 500
    result = lean("import examples.monte_carlo.proofs.MonteCarloProofs\n", "MonteCarloProofs.lean")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize("n", [1, 3, 7])
def test_new_jaxpr_uses_elementary_rules(n):
    def sample(key):
        return jax.random.randint(key, (), -2, 5).astype(jnp.float32)

    def estimator(key):
        return monte_carlo(key, sample, lambda x: x * x + 2.0, n)

    # Translate the IR itself, with an outer jit; no Python inspection is needed.
    closed = jax.make_jaxpr(jax.jit(estimator))(jax.random.key(0))
    source = translate(closed, name="estimator", random_model="uniform", readable=True)
    assert f"Rand.iid {n}" in source
    assert "KEY SPECIFICATION" in source
    source += f"""
example : (Generated.estimator (R := ℝ)).variance =
    (JaxLean.Rand.map (fun x => x * x + 2)
      (JaxLean.Rand.uniformInt (R := ℝ) (-2) 7 (by decide))).variance / {n} := by
  simp only [Generated.estimator, JaxLean.Tensor.map,
    JaxLean.Tensor.map₂, JaxLean.Tensor.sumFirst, JaxLean.Batch.reduceSum, JaxLean.Tensor.scalar]
  rw [JaxLean.Rand.variance_map_div,
    JaxLean.Rand.variance_iid_sum (f := fun _ v => v * v + 2)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  norm_num
  <;> ring
"""
    result = lean(source, f"MonteCarlo{n}.lean")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_repeating_one_draw_is_not_independent():
    def repeated(key):
        return jnp.mean(jnp.broadcast_to(uniform_six(key), (4,)))

    source = transpile(repeated, jax.random.key(0), random_model="uniform")
    assert "Rand.iid" not in source
    source += """
example : (Generated.repeated (R := ℝ)).variance = 35 / 12 := by
  unfold Generated.repeated
  dsimp [JaxLean.Tensor.sumFirst, JaxLean.Batch.reduceSum, JaxLean.Tensor.scalar, JaxLean.Tensor.reindex, JaxLean.Rand.map,
    JaxLean.Rand.variance, JaxLean.Rand.uniformInt, JaxLean.FiniteLaw.uniform,
    JaxLean.FiniteLaw.variance, JaxLean.FiniteLaw.mean]
  norm_num [Fin.sum_univ_succ]
"""
    result = lean(source, "RepeatedSingleDraw.lean")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", ["zero", "two_splits", "parent", "reuse", "vector", "dynamic", "nested"])
def test_unsupported_random_shapes_and_key_use(case):
    def bad(key, bound):
        if case == "vector":
            return jnp.mean(jax.random.randint(key, (4,), 0, 6).astype(jnp.float32))
        keys = jax.random.split(key, 0 if case == "zero" else 4)
        if case == "two_splits":
            more = jax.random.split(key, 4)
            return jnp.mean(jax.vmap(uniform_six)(keys) + jax.vmap(uniform_six)(more))
        if case == "parent":
            return uniform_six(key) + jnp.mean(jax.vmap(uniform_six)(keys))
        if case == "reuse":
            return jnp.mean(jax.vmap(uniform_six)(keys) + jax.vmap(uniform_six)(keys))
        if case == "dynamic":
            return jnp.mean(jax.vmap(lambda k: jax.random.randint(k, (), 0, bound).astype(jnp.float32))(keys))
        if case == "nested":
            return jnp.mean(jax.vmap(lambda k: monte_carlo(k, uniform_six, lambda x: x, 2))(keys))
        return jnp.mean(jax.vmap(uniform_six)(keys))

    with pytest.raises(TranslationError):
        transpile(bad, jax.random.key(0), jnp.float32(6), random_model="uniform")


def test_input_key_batch_is_not_assumed_independent():
    with pytest.raises(TranslationError, match="scalar root key"):
        transpile(lambda keys: jnp.mean(jax.vmap(uniform_six)(keys)),
                  jax.random.split(jax.random.key(0), 4), random_model="uniform")


def test_counterfeit_batched_randint_is_rejected():
    @jax.jit
    def _randint(key, lo, hi):
        return jax.random.randint(key, (), lo, hi) + 1

    def fake(key):
        return jnp.mean(jax.vmap(lambda k: _randint(k, 0, 6))(
            jax.random.split(key, 4)).astype(jnp.float32))

    with pytest.raises(TranslationError, match="does not match"):
        transpile(fake, jax.random.key(0), random_model="uniform")


@pytest.mark.lean
def test_wrong_variance_reduction_fails():
    # Averaging four samples divides variance by four, not by sixteen.
    source = """import examples.monte_carlo.proofs.MonteCarloProofs
import examples.random_program.proofs.RandomProgramProofs
open JaxLean JaxLean.Generated
example : (monte_carlo_4 (R := ℝ)).variance =
    (sample_times_100 (R := ℝ)).variance / 16 := by
  rw [MonteCarloProofs.monte_carlo_4_variance,
    RandomProgramProofs.sample_times_100_variance_value]
  norm_num
"""
    result = lean(source, "WrongMonteCarloVariance.lean")
    assert result.returncode != 0
    assert "unsolved goals" in result.stdout


@pytest.mark.lean
def test_weighted_sum_propagates_elementary_rules():
    def weighted(key, weights):
        draws = jax.vmap(uniform_six)(jax.random.split(key, 3))
        return jnp.sum(draws * weights) + 7.0

    source = transpile(weighted, jax.random.key(0), jnp.ones(3), random_model="uniform")
    source += """
example (weights : JaxLean.Tensor ℝ [3]) :
    (Generated.weighted weights).variance =
      ∑ i : Fin 3, (weights (i, ())) ^ 2 *
        (JaxLean.Rand.uniformInt (R := ℝ) 0 6 (by decide)).variance := by
  simp only [Generated.weighted, JaxLean.Tensor.sumFirst, JaxLean.Batch.reduceSum, JaxLean.Tensor.scalar, JaxLean.Tensor.map₂]
  rw [JaxLean.Rand.variance_map_add_const,
    JaxLean.Rand.variance_iid_sum (f := fun i v => v * weights (i, ()))]
  simp only [JaxLean.Rand.variance_map_mul]
"""
    result = lean(source, "WeightedMoments.lean")
    assert result.returncode == 0, result.stdout + result.stderr
