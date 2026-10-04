"""Kernel-checked equality to imported Jaxpr, including deliberate corruptions."""
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from jaxlean import certify, translate, TranslationError
from test_random_translation import lean, ROOT


def mean(x):
    return jnp.mean(x)


@pytest.mark.lean
def test_mean_certificate_and_reproducibility():
    subprocess.run([sys.executable, '-m', 'examples.certify', '--check'], cwd=ROOT, check=True)
    source = certify(jax.make_jaxpr(mean)(jnp.ones(2)), name='mean')
    assert 'Jaxpr.Program [[2]] []' in source
    assert 'mean_translation_correct' in source
    assert 'Tensor.sumFirst' in source
    result = lean(source, 'CertifiedMean.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize('n', [0, 1, 5])
def test_sequence_broadcast_and_multiple_inputs(n):
    def f(x, y, a):
        z = x * y + a
        return jnp.sum(z, axis=0) / 3.0
    closed = jax.make_jaxpr(f)(jnp.ones((n, 2)), jnp.ones((n, 2)), jnp.float32(0.1))
    source = certify(closed, name='sequence')
    assert '.broadcast [' in source
    result = lean(source, f'CertifiedSequence{n}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_explicit_broadcast_and_exact_float_literal():
    def f(x):
        return jnp.broadcast_to(x + jnp.float32(0.1), (2, 3))
    source = certify(jax.make_jaxpr(f)(jnp.float32(2)), name='broadcast')
    assert '.broadcast [2, 3]' in source
    # Stored float32 0.1, not decimal 1/10.
    assert '13421773' in source
    result = lean(source, 'CertifiedBroadcast.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_wrong_generated_divisor_fails_certificate():
    source = certify(jax.make_jaxpr(mean)(jnp.ones(2)), name='mean')
    # Change ONLY the generated function; the imported Jaxpr still divides by two.
    old = 'Tensor.scalar (2 : R)'
    assert source.count(old) == 1
    source = source.replace(old, 'Tensor.scalar (3 : R)')
    result = lean(source, 'CorruptedMean.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_wrong_generated_operation_fails_certificate():
    closed = jax.make_jaxpr(lambda x, y: x * y)(jnp.ones(3), jnp.ones(3))
    source = certify(closed, name='multiply')
    assert 'a0 * a1' in source
    source = source.replace('a0 * a1', 'a0 + a1')
    result = lean(source, 'CorruptedMultiply.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_bad_variable_reference_fails_lean_typechecking():
    source = certify(jax.make_jaxpr(lambda x: x)(jnp.ones(3)), name='identity')
    source = source.replace('.ret (.var .here)', '.ret (.var (.there .here))')
    result = lean(source, 'CorruptedReference.lean')
    assert result.returncode != 0


@pytest.mark.parametrize('fn,args', [
    (lambda x: (x, x), (jnp.ones(3),)),
    (jax.jit(lambda x: x + 1.0), (jnp.ones(3),)),
])
def test_supported_translation_is_not_silently_certified(fn, args):
    jp = jax.make_jaxpr(fn)(*args)
    assert translate(jp)  # Ordinary support is deliberately broader.
    with pytest.raises(TranslationError):
        certify(jp)


def test_captured_array_constants_fail_closed():
    a = jnp.array([1.0, 2.0])
    jp = jax.make_jaxpr(lambda x: x + a)(jnp.ones(2))
    with pytest.raises(TranslationError, match='captured'):
        certify(jp)


@pytest.mark.lean
@pytest.mark.parametrize('fn,args', [
    (lambda: jnp.float32(-0.1), ()),
    (lambda x: x, (jnp.ones((2, 3)),)),
    (lambda x, y: y, (jnp.ones(2), jnp.ones(3))),
])
def test_literal_identity_and_older_environment_reference(fn, args):
    source = certify(jax.make_jaxpr(fn)(*args), name='returns')
    result = lean(source, f'CertifiedReturn{len(args)}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_wrong_same_shape_operand_fails_certificate():
    source = certify(jax.make_jaxpr(lambda x, y: x + y)(jnp.ones(3), jnp.ones(3)), name='add')
    assert ' x0 x1' in source
    # Well typed, but now computes x+x instead of x+y. The IR remains unchanged.
    source = source.replace(' x0 x1\n', ' x0 x0\n', 1)
    result = lean(source, 'CorruptedOperand.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_extended_pointwise_rules():
    def f(x, y):
        a = jnp.abs(-x + y) ** 3
        return jnp.maximum(jnp.minimum(a, (x - y) ** 2), 0.0)
    source = certify(jax.make_jaxpr(f)(jnp.ones(3), jnp.ones(3)), name='pointwise')
    result = lean(source, 'CertifiedPointwise.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_negative_integer_power():
    source = certify(jax.make_jaxpr(lambda x: x ** -2)(jnp.ones(3)), name='inverse_square')
    result = lean(source, 'CertifiedNegativePower.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize('n', [0, 1, 4])
def test_certified_dense_layer(n):
    from examples.certify import relu_layer
    source = certify(jax.make_jaxpr(relu_layer)(jnp.ones((n, 2)), jnp.ones((2, 3)), jnp.ones(3)), name='layer')
    assert '.matmul' in source and '.prepend' in source
    if n != 1:
        assert '.expandFirst' in source
    result = lean(source, f'CertifiedLayer{n}.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_transpose_and_matrix_products():
    from examples.certify import gram
    source = certify(jax.make_jaxpr(gram)(jnp.ones((3, 2))), name='gram')
    assert '.transpose2' in source
    result = lean(source, 'CertifiedGram.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    source = certify(jax.make_jaxpr(lambda x, w: x @ w)(jnp.ones(2), jnp.ones((2, 3))), name='vecmat')
    assert '.vecmat' in source
    result = lean(source, 'CertifiedVecmat.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_float_cast_certificate_is_explicitly_real_semantics():
    def f(x):
        return jax.lax.stop_gradient(x.astype(jnp.float16)).astype(jnp.float32)
    x = jnp.array([1.0001], dtype=jnp.float32)
    assert float(f(x)[0]) != float(x[0])  # Machine rounding is observably different.
    source = certify(jax.make_jaxpr(f)(x), name='casts')
    assert '.copy' in source
    result = lean(source, 'CertifiedRealCasts.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_wrong_matrix_operand_order_fails_certificate():
    source = certify(jax.make_jaxpr(lambda x, y: x @ y)(jnp.ones((2, 2)), jnp.ones((2, 2))), name='matmul')
    assert 'Tensor.matmul x0 x1' in source
    source = source.replace('Tensor.matmul x0 x1', 'Tensor.matmul x1 x0', 1)
    result = lean(source, 'CorruptedMatmul.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.lean
def test_wrong_transpose_permutation_fails_certificate():
    source = certify(jax.make_jaxpr(lambda x: x.T)(jnp.ones((2, 2))), name='transpose')
    assert '(i.2.1, i.1, ())' in source
    source = source.replace('(i.2.1, i.1, ())', '(i.1, i.2.1, ())', 1)
    result = lean(source, 'CorruptedTranspose.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.parametrize('fn,args', [
    (lambda x: x.astype(jnp.int32), (jnp.ones(3),)),
])
def test_new_certificate_boundaries_fail_closed(fn, args):
    with pytest.raises(TranslationError):
        certify(jax.make_jaxpr(fn)(*args))
