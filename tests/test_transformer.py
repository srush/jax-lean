"""Function-boundary certificates and transformer properties, including mutations."""
from fractions import Fraction
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxlean import certify, certify_module, TranslationError
from examples.transformer.code import transformer, attention, normalize
from examples.transformer.generate import source
from test_random_translation import lean, ROOT
from test_translation import lean_input


@pytest.mark.lean
def test_generated_module_and_boundary_proofs():
    subprocess.run([sys.executable, '-m', 'examples.transformer.generate', '--check'],
                   cwd=ROOT, check=True)
    code = source()
    assert code.count('def «project»') == 1  # Repeated calls share one certificate.
    assert code.count('def «transformer_block»') == 1
    assert '.call transformer_block_ir' in code
    assert ('jaxpr_certificate [_root_.«JaxLean».«TransformerJax».transformer_ir, '
            '_root_.«JaxLean».«TransformerJax».«transformer», '
            '_root_.«JaxLean».«TransformerJax».transformer_block_translation_correct]') in code
    assert '(«w0» : Tensor ℝ [2, 2])' in code
    result = lean('''import examples.transformer.proofs.TransformerProofs
#print axioms JaxLean.TransformerJax.certified_transformer_permute
#print axioms JaxLean.TransformerJax.normalize_sum_one
''', 'TransformerAudit.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'sorryAx' not in result.stdout


@pytest.mark.lean
def test_jax_and_lean_transformer_agree():
    x = jnp.array([[1., -1.], [0., 2.], [2., 1.]])
    weights = tuple(jnp.array(a, dtype=jnp.float32) for a in (
        [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, 1]],
        [[1, -1], [0, 1]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 0]]))
    call = 'JaxLean.TransformerJax.transformer (R := ℚ) ' + ' '.join(
        lean_input(a) for a in (x, *weights))
    code = '''import examples.transformer.generated.Transformer
open JaxLean
'''+f'''#eval IO.println (Lean.Json.compress (Lean.toJson
  ((Tensor.toList ({call})).map (fun x => s!"{{x.num}}/{{x.den}}"))))
'''
    result = lean(code, 'TransformerValues.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    actual = np.array([float(Fraction(v)) for v in json.loads(result.stdout.strip())])
    expected = np.asarray(transformer(x, *weights)).flatten()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    perm = jnp.array([2, 0, 1])
    np.testing.assert_allclose(transformer(x[perm], *weights), transformer(x, *weights)[perm],
                               rtol=2e-5, atol=2e-5)


def test_normalization_and_selection_boundary():
    scores = jnp.array([[-2., 1., 3.], [0., 0., 0.], [-1., -2., -3.]])
    values = normalize(scores)
    assert np.all(values >= 0)
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    # Attention is not equivariant to arbitrary duplicate/drop selections of tokens.
    x = jnp.array([[1., 0.], [0., 2.], [3., 1.]])
    selection = jnp.array([0, 0, 1])
    assert not np.allclose(attention(x[selection], x[selection], x[selection]),
                           attention(x, x, x)[selection])


@pytest.mark.lean
@pytest.mark.parametrize('shape', [(0, 3), (2, 0), (1, 1), (2, 3)])
def test_row_reduction_and_broadcast_certificates(shape):
    def fn(x):
        return x / jnp.sum(1.0 + x * x, axis=1, keepdims=True)
    code = certify(jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32)))
    result = lean(code, f'RowReduction{shape[0]}x{shape[1]}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_well_typed_wrong_callee_fails_certificate():
    code = source()
    original = '«project» (R := R) (call_forward_result) («wq»)'
    assert code.count(original) == 1
    code = code.replace(original, '«forward» (R := R) (call_forward_result) («wq»)')
    result = lean(code, 'WrongTransformerCallee.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_wrong_key_projection_fails_certificate():
    code = source()
    original = '«project» (R := R) (call_forward_result) («wk»)'
    assert code.count(original) == 1
    code = code.replace(original, '«project» (R := R) (call_forward_result) («wq»)')
    result = lean(code, 'WrongTransformerKey.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_module_name_collisions_and_distinct_call_bodies():
    @jax.jit
    def f(x):
        return x * 2.0
    first = f
    @jax.jit
    def f(x):
        return x + 3.0
    jp = jax.make_jaxpr(lambda x: first(f(x)))(jnp.ones(2))
    code = certify_module(jp, name='f')
    assert 'def «f_2»' in code and 'def «f_3»' in code
    result = lean(code, 'DistinctCallBodies.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('fn', [
    jax.jit(lambda x: (x, x)),
    jax.jit(lambda x: jnp.floor(x)),
    jax.jit(lambda x: x + jnp.array([1., 2.])),
])
def test_unsupported_children_fail_closed(fn):
    with pytest.raises(TranslationError):
        certify_module(jax.make_jaxpr(fn)(jnp.ones(2)))


@pytest.mark.lean
@pytest.mark.parametrize('tokens,hidden', [(1, 2), (4, 3)])
def test_other_transformer_shapes_certify(tokens, hidden):
    x = jax.ShapeDtypeStruct((tokens, hidden), jnp.float32)
    w = jax.ShapeDtypeStruct((hidden, hidden), jnp.float32)
    code = certify_module(jax.make_jaxpr(transformer)(x, *([w] * 8)), name='network')
    result = lean(code, f'TransformerShape{tokens}x{hidden}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_explicit_trailing_broadcast_certificate():
    jp = jax.make_jaxpr(lambda x: jnp.broadcast_to(x, (2, 3)))(jnp.ones((2, 1)))
    code = certify(jp)
    assert '.broadcast_in_dim [2, 3] [0, 1]' in code
    result = lean(code, 'ExplicitTrailingBroadcast.lean')
    assert result.returncode == 0, result.stdout + result.stderr
