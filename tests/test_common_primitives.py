"""Fast coverage checks; Lean integration is explicit and batched."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxlean import certify_module, translate, TranslationError
from examples.common_primitives import code as common
from test_random_translation import lean


def array(*s, dtype=jnp.float32):
    return jax.ShapeDtypeStruct(s, dtype)


CASES = [
    (lambda c, x, y: jax.lax.select(c, x, y),
     (array(dtype=jnp.bool_), array(3), array(3))),
    (lambda a, b: jax.lax.bitwise_and(a, b),
     (array(dtype=jnp.bool_), array(3, dtype=jnp.bool_))),
    (common.masked_values, (array(3), array(3, dtype=jnp.bool_))),
    (common.batched_scores, (array(2, 3, 4), array(2, 5, 4))),
    (common.row_softmax, (array(2, 3),)),
    (common.embeddings, (array(5, 3), array(2, dtype=jnp.int32))),
    (common.shifted_embeddings, (array(5, 3), array(2, dtype=jnp.int32))),
    (common.rearrange, (array(2, 1, 3), array(2, 2, 3))),
    (common.extrema, (array(2, 3, 4),)),
    (lambda x: jnp.sin(x) + jnp.sqrt(x), (array(3),)),
    (lambda x: jnp.sum(x, axis=1), (array(2, 3, 4),)),
    (lambda x: jnp.sum(x, axis=(0, 2)), (array(2, 3, 4),)),
    (lambda x: jax.lax.broadcast_in_dim(x, (3, 2), (0,)), (array(3),)),
    (lambda x: x.transpose(2, 1, 0), (array(2, 3, 4),)),
    (lambda x: x[::-1, ::-1], (array(2, 3),)),
    (lambda x: jax.lax.reshape(x, (6,), (1, 0)), (array(2, 3),)),
    (lambda x: jnp.concatenate([x, x, x], axis=0), (array(2, 3),)),
    (lambda x: x > 0, (array(3),)),
    (lambda x: x + jnp.int32(1), (array(2, dtype=jnp.int32),)),
    (lambda x: jnp.where(x < 0, x + jnp.int32(5), x), (array(2, dtype=jnp.int32),)),
    (lambda x: x.astype(jnp.float32), (array(2, dtype=jnp.int32),)),
    (lambda x: jax.nn.softmax(x, axis=(0, 2)), (array(2, 3, 4),)),
]


@pytest.mark.parametrize('fn,args', CASES)
def test_common_primitives_have_certificate_rules(fn, args):
    source = certify_module(jax.make_jaxpr(fn)(*args))
    assert '_translation_correct' in source
    assert 'sorry' not in source


@pytest.mark.parametrize('fn,args,message', [
    (lambda x, i: x[i], (array(5, 3), array(2, dtype=jnp.int32)), 'clip'),
    (lambda x, i: jnp.take(x, i, axis=0, mode='fill'), (array(5, 3), array(2, dtype=jnp.int32)), 'clip'),
    (lambda x: jax.lax.reduce_max(x, (1,)), (array(2, 0),), 'empty extrema'),
    (lambda x: jax.lax.reduce_min(x, (1,)), (array(2, 0),), 'empty extrema'),
    (lambda x: x + jnp.inf, (array(3),), 'infinity'),
    (lambda x: x.astype(jnp.int32), (array(3),), 'conversion'),
    (lambda x: x + 1, (array(3, dtype=jnp.uint32),), 'int32'),
])
def test_new_boundaries_fail_closed(fn, args, message):
    with pytest.raises(TranslationError, match=message):
        certify_module(jax.make_jaxpr(fn)(*args))


@pytest.mark.lean
def test_common_certificates_check_together():
    imports, bodies = set(), []
    for i, (fn, args) in enumerate(CASES):
        lines = certify_module(jax.make_jaxpr(fn)(*args), namespace=f'Case{i}').splitlines()
        imports.update(line for line in lines if line.startswith('import '))
        bodies.extend(line for line in lines if not line.startswith('import '))
    result = lean('\n'.join([*sorted(imports), *bodies]), 'CommonCertificates.lean')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'declaration uses `sorry`' not in result.stdout


@pytest.mark.lean
def test_clipped_gather_and_int32_overflow_match_jax():
    table = np.array([[10*i+j for j in range(3)] for i in range(5)], dtype=np.float32)
    ids = jnp.array([-1, 100], dtype=jnp.int32)
    actual = common.embeddings(jnp.asarray(table), ids)
    overflow = common.shifted_embeddings(jnp.asarray(table), jnp.array([2147483647, -1], dtype=jnp.int32))
    source = '''import examples.common_primitives.generated.CommonEmbeddings
import examples.common_primitives.generated.CommonShifted
open JaxLean
open JaxLean.CommonJax
private def table : Tensor ℚ [5, 3] := fun i => 10 * i.1.val + i.2.1.val
private def ids : Tensor Int32 [2] := fun i => if i.1 == 0 then -1 else 100
private def overflowIds : Tensor Int32 [2] := fun i => if i.1 == 0 then 2147483647 else -1
'''
    for i in range(2):
        for j in range(3):
            source += f'#guard Embeddings.embeddings table ids ({i}, {j}, ()) == {int(actual[i,j])}\n'
            source += f'#guard Shifted.shifted_embeddings table overflowIds ({i}, {j}, ()) == {int(overflow[i,j])}\n'
    result = lean(source, 'ClippedGatherValues.lean')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.lean
def test_wrong_selection_branch_fails_certificate():
    source = certify_module(jax.make_jaxpr(lambda x: jnp.where(x > 0, x, -x))(jnp.ones(3)))
    assert ' then ' in source
    # Corrupt only the emitted function's conditional, leaving imported IR intact.
    import re
    source, n = re.subn(r'if (.*?) then (.*?) else ([^\n]+)', r'if \1 then \3 else \2', source, count=1)
    assert n == 1
    result = lean(source, 'WrongTypedSelect.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout
