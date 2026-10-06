"""Named SSA syntax must preserve the typed de Bruijn references exactly."""
import jax
import jax.numpy as jnp
import pytest

from jaxlean import certify_module
from jaxlean.importers.jaxpr import JaxprImporter


def test_names_are_stable_and_unbounded():
    assert [JaxprImporter.variable_name(i) for i in (0, 1, 25, 26, 27, 701, 702)] == [
        'a', 'b', 'z', 'v26', 'v27', 'v701', 'v702']
    closed = jax.make_jaxpr(lambda a, b: (a + b) * a)(jnp.ones(3), jnp.ones(3))
    source = certify_module(closed, named_vars=True)
    assert 'jaxpr% (a : (.real, [3]), b : (.real, [3]))' in source
    assert 'c : (.real, [3]) := .add a b' in source
    assert 'd : (.real, [3]) := .mul c a' in source
    assert '.here' not in source and '.there' not in source
    assert 'let ' not in JaxprImporter(closed).run()


@pytest.mark.lean
@pytest.mark.parametrize('fn', [
    lambda a, b: (a + b) * a + b,
    lambda a, b: jnp.where(a > b, a + b, a),
])
def test_named_syntax_is_definitionally_equal_to_positional_ir(fn):
    from test_random_translation import lean
    closed = jax.make_jaxpr(fn)(jnp.ones(3), jnp.ones(3))
    sources = [certify_module(closed, namespace=ns, named_vars=named)
               for ns, named in [('Positional', False), ('Named', True)]]
    lines = '\n'.join(sources).splitlines()
    imports = list(dict.fromkeys(line for line in lines if line.startswith('import ')))
    body = [line for line in lines if not line.startswith('import ')]
    source = '\n'.join(imports + body) + '\nexample : Named.program_ir = Positional.program_ir := rfl\n'
    result = lean(source, 'NamedVariables.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
@pytest.mark.parametrize('body, message', [
    ('c : (.real, [3]) := .add a missing; return c', 'Unknown identifier'),
    ('a : (.real, [3]) := .add a a; return a', 'duplicate SSA variable'),
    ('c : (.bool, [3]) := .add a a; return c', 'mismatch'),
    ('c : (.real, [2]) := .neg a; return a', 'mismatch'),
])
def test_named_syntax_rejects_invalid_programs(body, message):
    from test_random_translation import lean
    source = """import JaxLean.Verification.Syntax
open JaxLean
example : Jaxpr.Program [(.real, [3])] (.real, [3]) :=
  jaxpr% (a : (.real, [3])) { """ + body + ' }\n'
    result = lean(source, 'InvalidNamedVariables.lean')
    assert result.returncode != 0
    assert message.lower() in (result.stdout + result.stderr).lower()
