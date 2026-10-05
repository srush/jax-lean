"""Bounds on actual transpiled vector programs, using reusable norm rules."""
import jax
import jax.numpy as jnp
import pytest
from jaxlean import transpile
from examples.norms.code import vector_norm, linear_clip, radial_clip, batch_radial_clip
from test_random_translation import lean


@pytest.mark.lean
def test_examples_execute_and_proofs_check():
    x = jnp.array([3.0, 4.0, 0.0])
    assert float(jax.jit(vector_norm)(x)) == 5.0
    assert jnp.allclose(jax.jit(radial_clip)(x, 2.0), jnp.array([1.2, 1.6, 0.0]))
    assert jnp.all(jax.jit(radial_clip)(jnp.zeros(3), 2.0) == 0)
    assert jax.jit(batch_radial_clip)(jnp.stack([x, x]), 2.0).shape == (2, 3)
    result = lean('import examples.norms.proofs.NormProofs\n', 'NormExamples.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize('n,m', [(0, 2), (1, 1), (5, 4)])
def test_general_rules_apply_to_new_shapes(n, m):
    source = 'import JaxLean.Stdlib\n' + transpile(
        linear_clip, jnp.ones(n), jnp.ones((n, m)), jnp.float32(1))
    assert 'Tensor.vecmat' in source
    source += f"""
example (x : Tensor ℝ [{n}]) (w : Tensor ℝ [{n}, {m}]) (r : ℝ) (hr : 0 ≤ r) :
    Tensor.vectorNorm (Generated.linear_clip x w (Tensor.scalar r)) ≤
      Tensor.frobeniusNorm w * Tensor.vectorNorm x := by
  change Tensor.vectorNorm (Tensor.map (Batch.clip r) (Tensor.vecmat x w)) ≤ _
  exact (Tensor.vectorNorm_clip_le r hr _).trans (Tensor.vectorNorm_vecmat_le x w)
"""
    result = lean(source, f'NormLinear{n}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize('n', [0, 1, 5])
def test_radial_clipping_new_shapes(n):
    source = 'import JaxLean.Stdlib\n' + transpile(radial_clip, jnp.ones(n), jnp.float32(1))
    source += f"""
example (x : Tensor ℝ [{n}]) (r : ℝ) (hr : 0 < r) :
    Tensor.vectorNorm (Generated.radial_clip x (Tensor.scalar r)) ≤
      min r (Tensor.vectorNorm x) := by
  simpa only [Generated.radial_clip, Tensor.vectorNorm, Tensor.map, Tensor.map₂,
    Tensor.scalar, Tensor.sumFirst, Batch.reduceSum, RealOps.sqrt, Batch.clipL2,
    Batch.l2, pow_two] using
    Batch.l2_clipL2_le r hr (fun i : Fin {n} => x (i, ()))
"""
    result = lean(source, f'NormRadial{n}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_component_clipping_does_not_imply_l2_ball():
    # Clip each of two coordinates to 1: the norm is sqrt(2), not at most 1.
    source = """import JaxLean.Stdlib
open JaxLean
example : Batch.l2 (Batch.vmap (Batch.clip 1) (fun _ : Fin 2 => (1 : ℝ))) ≤ 1 := by
  norm_num [Batch.l2, Batch.vmap, Batch.clip, Real.sqrt_le_iff]
"""
    result = lean(source, 'WrongClipRadius.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_linear_map_can_increase_norm():
    source = """import JaxLean.Stdlib
open JaxLean
example : Batch.l2 (fun _ : Fin 1 => (2 : ℝ)) ≤
    Batch.l2 (fun _ : Fin 1 => (1 : ℝ)) := by
  norm_num [Batch.l2]
"""
    result = lean(source, 'WrongLinearContraction.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout
