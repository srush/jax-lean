"""General batch algebra applied to ordinary deterministic Jaxpr."""
import jax
import jax.numpy as jnp
import pytest
from jaxlean import transpile
from test_random_translation import lean


@pytest.mark.lean
@pytest.mark.parametrize("n", [0, 1, 4])
def test_linear_rule_on_generated_reduction(n):
    def scaled_sum(x, scale):
        return jnp.sum(jax.vmap(lambda row: row * scale)(x), axis=0)

    source = 'import JaxLean.Stdlib\n' + transpile(
        scaled_sum, jnp.ones((n, 2)), jnp.float32(3))
    assert 'Tensor.sumFirst' in source
    source += f"""
example (x : Tensor ℝ [{n}, 2]) (scale : Tensor ℝ []) :
    Generated.scaled_sum x scale =
      Tensor.map (fun v => v * scale ()) (Tensor.sumFirst x) := by
  exact Tensor.sumFirst_map_linear (LinearMap.mulRight ℝ (scale ())) x
"""
    result = lean(source, f'BatchLinear{n}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_matrix_rule_on_generated_vmap():
    def batch_matmul_sum(x, weights):
        return jnp.sum(jax.vmap(lambda row: row @ weights)(x), axis=0)

    source = 'import JaxLean.Stdlib\n' + transpile(
        batch_matmul_sum, jnp.ones((3, 2)), jnp.ones((2, 4)))
    assert 'Tensor.sumFirst' in source
    assert 'Tensor.matmul' in source
    source += """
example (x : Tensor ℝ [3, 2]) (w : Tensor ℝ [2, 4]) :
    Generated.batch_matmul_sum x w =
      fun j => ∑ a : Fin 2, Tensor.sumFirst x (a, ()) * w (a, j) := by
  exact Tensor.sumFirst_matmul x w
"""
    result = lean(source, 'BatchMatmul.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_bilinear_cross_terms_and_general_expectation():
    source = """import JaxLean.Stdlib
import Mathlib.Algebra.Algebra.Bilinear
open JaxLean
open scoped BigOperators
example (x : Fin 3 → ℝ) (y : Fin 4 → ℝ) :
    Batch.reduceSum x * Batch.reduceSum y =
      Batch.reduceSum (fun i => Batch.reduceSum (fun j => x i * y j)) := by
  exact Batch.bilinear_reduceSum_both (LinearMap.mul ℝ ℝ) x y

-- No independence: all three summands can reuse the same underlying value.
example (x : Rand ℝ) :
    (Rand.map (fun v => ∑ _ : Fin 3, v) x).mean =
      ∑ _ : Fin 3, (Rand.map id x).mean := by
  exact Rand.mean_map_sum x (fun _ v => v)
"""
    result = lean(source, 'BatchBilinear.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_bilinear_diagonal_shortcut_is_false():
    source = """import JaxLean.Stdlib
open JaxLean
example : Batch.reduceSum (fun _ : Fin 2 => (1 : ℚ)) *
    Batch.reduceSum (fun _ : Fin 2 => (1 : ℚ)) =
    Batch.reduceSum (fun _ : Fin 2 => (1 : ℚ) * 1) := by
  norm_num [Batch.reduceSum]
"""
    result = lean(source, 'WrongBilinearDiagonal.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_independence_bridge_without_rand():
    source = """import JaxLean.Stdlib.BatchProbability
import Mathlib.MeasureTheory.Function.SpecialFunctions.Basic
open MeasureTheory ProbabilityTheory JaxLean
example {Ω : Type} [MeasurableSpace Ω] {μ : Measure Ω}
    (X : Fin 4 → Ω → ℝ) (hX : iIndepFun X μ) (a : Fin 4 → ℝ) :
    iIndepFun (fun i ω => X i ω * a i) μ := by
  exact Batch.independent_vmap hX (fun i v => v * a i)
    (fun i => measurable_id.mul_const (a i))
"""
    result = lean(source, 'BatchIndependence.lean')
    assert result.returncode == 0, result.stdout + result.stderr
