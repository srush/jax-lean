"""Randint estimators and independent pseudocode contracts, not just snapshots."""
from fractions import Fraction
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxlean import certify, TranslationError
from examples import randint_monte_carlo as mc
from examples import tensor_puzzles as puzzles
from examples.certify_more import artifacts
from test_random_translation import lean, ROOT
from test_translation import lean_input


def test_generation_and_proof_axioms():
    subprocess.run([sys.executable, '-m', 'examples.certify_more', '--check'], cwd=ROOT, check=True)
    result = lean('''import JaxLean.RandintMonteCarloProofs
import JaxLean.TensorPuzzleProofs
#print axioms JaxLean.MCJax.certified_die_variance
#print axioms JaxLean.MCJax.certified_grid_variance
#print axioms JaxLean.PuzzleJax.certified_sum
#print axioms JaxLean.PuzzleJax.certified_flip
#print axioms JaxLean.PuzzleJax.certified_outer_flatten
''', 'MoreExamplesAxioms.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'sorryAx' not in result.stdout


def test_actual_randint_outputs_pass_through_lean_kernels():
    key = jax.random.key(18)
    die_draws = jax.vmap(mc.die_draw)(jax.random.split(key, 16))
    grid_draws = jax.vmap(mc.grid_index)(jax.random.split(key, 8))
    assert set(np.asarray(die_draws)).issubset(set(range(1, 7)))
    assert set(np.asarray(grid_draws)).issubset(set(range(4)))
    code = 'import JaxLean.Generated.DieEstimate\nimport JaxLean.Generated.GridEstimate\nopen JaxLean\n'
    for name, draws in [('die_estimate', die_draws), ('grid_estimate', grid_draws)]:
        call = f'MCJax.{name} (R := ℚ) {lean_input(draws)} ()'
        code += f'#eval IO.println (let x := {call}; s!"{{x.num}}/{{x.den}}")\n'
    result = lean(code, 'MonteCarloRealSamples.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    values = [float(Fraction(line)) for line in result.stdout.splitlines()]
    np.testing.assert_allclose(values, [mc.mc_die_16(key), mc.mc_grid_square_8(key)], rtol=1e-6)


def test_puzzle_jax_lean_and_loop_specs_agree():
    a = jnp.array([2., -3.])
    b = jnp.array([4., 0., -2.])
    x = jnp.array([2., -3., 4., -5.])
    code = '''import JaxLean.Generated.PuzzleSum
import JaxLean.Generated.PuzzleFlip
import JaxLean.Generated.PuzzleOuterFlatten
import JaxLean.Pseudocode
open JaxLean
'''
    calls = [
        f'PuzzleJax.sum (R := ℚ) {lean_input(x)}',
        f'PuzzleJax.flip (R := ℚ) {lean_input(x)}',
        f'PuzzleJax.outer_flatten (R := ℚ) {lean_input(a)} {lean_input(b)}',
        f'Tensor.scalar (Pseudocode.sum {lean_input(x)})',
        f'Pseudocode.flip {lean_input(x)}',
        f'Pseudocode.flatten (Pseudocode.outer {lean_input(a)} {lean_input(b)})',
    ]
    for call in calls:
        code += f'''#eval IO.println (Lean.Json.compress (Lean.toJson
            ((Tensor.toList ({call})).map (fun x => s!"{{x.num}}/{{x.den}}"))))
'''
    result = lean(code, 'PuzzleThreeWay.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    values = [[float(Fraction(v)) for v in json.loads(line)] for line in result.stdout.splitlines()]
    expected = [np.asarray(puzzles.puzzle_sum(x)).reshape(-1), np.asarray(puzzles.puzzle_flip(x)),
                np.asarray(puzzles.outer_flatten(a, b))]
    for actual, reference in zip(values, expected * 2, strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize('n', [0, 1, 5])
def test_vector_dot_and_reverse_certificates(n):
    for label, fn in [('dot', lambda x: x @ jnp.ones_like(x)), ('reverse', lambda x: x[::-1])]:
        code = certify(jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((n,), jnp.float32)))
        result = lean(code, f'Puzzle{label}{n}.lean')
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('shape,target', [((0, 3), (0,)), ((1, 1), (1,)), ((2, 3, 4), (4, 6))])
def test_reshape_certificate_sizes(shape, target):
    code = certify(jax.make_jaxpr(lambda x: x.reshape(target))(jax.ShapeDtypeStruct(shape, jnp.float32)))
    result = lean(code, f'Reshape{len(shape)}_{np.prod(shape)}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


def test_wrong_reversal_fails_certificate():
    code = dict(artifacts())['PuzzleFlip']
    assert '(i.1.rev, ())' in code
    code = code.replace('(i.1.rev, ())', '(i.1, ())', 1)
    result = lean(code, 'WrongPuzzleFlip.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


def test_wrong_flatten_order_fails_certificate():
    code = dict(artifacts())['PuzzleOuterFlatten']
    original = 'Tensor.reshape (t := [6]) (by decide) («a»)'
    replacement = ('Tensor.reshape (t := [6]) (by decide) '
                   '(Tensor.reindex (s := [2, 3]) (t := [3, 2]) '
                   '(fun i => (i.2.1, i.1, ())) «a»)')
    assert original in code
    result = lean(code.replace(original, replacement, 1), 'WrongPuzzleFlatten.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


def test_wrong_monte_carlo_variance_is_rejected():
    result = lean('''import JaxLean.RandintMonteCarloProofs
open JaxLean.MCJax
example : (mc_grid_square_8 (R := ℝ)).variance = 49 / 1024 := by
  rw [mc_grid_variance]
  norm_num
''', 'WrongMCVariance.lean')
    assert result.returncode != 0


def test_rank_two_reverse_still_rejected():
    with pytest.raises(TranslationError, match='rev'):
        certify(jax.make_jaxpr(lambda x: x[::-1])(jnp.ones((2, 3))))
