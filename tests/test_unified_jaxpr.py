"""The imported graph uses one IR, preserving source operations and bindings."""
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import pytest

from jaxlean import certify_module


def ir_bodies(source):
    return re.findall(r'def (\w+)_ir : Jaxpr.Program[^\n]* :=\n(.*?)\n\n', source, re.S)


def eye(n):
    i = jnp.arange(n, dtype=jnp.float32)
    return jnp.where(i[:, None] == i[None, :], 1., 0.)


def test_eye_preserves_equations_and_broadcast_parameters():
    source = certify_module(jax.make_jaxpr(eye, static_argnums=(0,))(3), name='eye')
    bodies = dict(ir_bodies(source))
    assert len(bodies) == 2
    body = bodies['eye']
    assert re.findall(r'\.bind \(\.(\w+)', body) == [
        'iota', 'broadcast_in_dim', 'broadcast_in_dim', 'eq']
    assert body.count('.call ') == 1
    assert '.broadcast_in_dim [3, 1] [0]' in body
    assert '.broadcast_in_dim [1, 3] [1]' in body
    assert '.reindex' not in body and '.real (' not in source
    assert 'TypedJaxpr' not in source
    child = next(value for name, value in bodies.items() if name != 'eye')
    assert re.findall(r'\.bind \(\.(\w+)', child) == [
        'broadcast_in_dim', 'broadcast_in_dim', 'select_n']


def test_mixed_and_real_calls_share_the_same_ir():
    @jax.jit
    def twice(x):
        return x * 2.

    def masked(x):
        return jnp.where(x > 0., twice(x), x)

    source = certify_module(jax.make_jaxpr(masked)(jnp.ones(3)), name='masked')
    bodies = dict(ir_bodies(source))
    assert '.call twice_ir' in bodies['masked']
    assert '.gt ' in bodies['masked']
    assert bodies['masked'].count('.bind ') == 1  # No invented scalar broadcast binding.
    assert 'TypedJaxpr' not in source and '.real (' not in source
    core = (Path(__file__).parents[1] / 'JaxLean/Core/Jaxpr.lean').read_text()
    assert core.count('inductive Program') == 1
    assert not (Path(__file__).parents[1] / 'JaxLean/Core/TypedJaxpr.lean').exists()


@pytest.mark.lean
def test_broadcast_parameter_corruption_fails_certificate():
    from test_random_translation import lean
    # Both axis choices are well typed; only the original matches the translation.
    fn = lambda x: jax.lax.broadcast_in_dim(x, (3, 3), (0,))
    source = certify_module(jax.make_jaxpr(fn)(jnp.ones(3)), name='broadcast')
    assert '.broadcast_in_dim [3, 3] [0]' in source
    good = lean(source, 'UnifiedBroadcast.lean')
    assert good.returncode == 0, good.stdout + good.stderr
    bad = lean(source.replace('.broadcast_in_dim [3, 3] [0]',
                              '.broadcast_in_dim [3, 3] [1]'), 'CorruptUnifiedBroadcast.lean')
    assert bad.returncode != 0 and 'unsolved goals' in bad.stdout


@pytest.mark.lean
def test_unified_ir_rejects_boolean_arithmetic():
    from test_random_translation import lean
    result = lean('''import JaxLean.Core.Jaxpr
open JaxLean
example : Jaxpr.Program [(.bool, [3])] (.bool, [3]) :=
  .bind (.add (.var .here) (.var .here) (t := [3])) <|
  .ret (.var .here)
''', 'InvalidBooleanAdd.lean')
    assert result.returncode != 0
    assert 'numeric' in result.stdout


def test_variadic_concat_and_static_scatter_keep_source_equation_count():
    from jaxlean import certify
    for fn, args in [
        (lambda x: jnp.concatenate([x, x, x]), (jnp.ones(3),)),
        (lambda x, u: x.at[jnp.array([2, 0])].set(u), (jnp.ones(4), jnp.ones(2))),
    ]:
        closed = jax.make_jaxpr(fn)(*args)
        body = dict(ir_bodies(certify(closed)))['program']
        ops = re.findall(r'\.bind \(\.(\w+)', body)
        assert ops == [eq.primitive.name.replace('-', '_') for eq in closed.jaxpr.eqns]
        assert '.reindex' not in body and '.set ' not in body
