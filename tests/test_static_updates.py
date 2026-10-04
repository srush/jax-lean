import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import tensor_puzzles as puzzles
from examples import scatter_updates
from jaxlean import certify, TranslationError
from jaxlean.static_index import integer_equation, scatter_plan
from test_random_translation import lean


@pytest.mark.parametrize('loop,vector,args', [
    (puzzles.loop_sum, puzzles.puzzle_sum, (jnp.arange(4, dtype=jnp.float32),)),
    (puzzles.loop_outer, puzzles.puzzle_outer, (jnp.arange(2, dtype=jnp.float32), jnp.arange(3, dtype=jnp.float32))),
    (puzzles.loop_flip, puzzles.puzzle_flip, (jnp.arange(4, dtype=jnp.float32),)),
    (puzzles.loop_flatten, puzzles.puzzle_flatten, (jnp.arange(6, dtype=jnp.float32).reshape(2, 3),)),
])
def test_jax_loops_and_vectorized_forms(loop, vector, args):
    np.testing.assert_array_equal(loop(*args), vector(*args))
    source = certify(jax.make_jaxpr(loop)(*args))
    assert 'Tensor.scatterSet' in source and '.set (s :=' in source


@pytest.mark.parametrize('fn', [
    lambda x: x.at[7].set(9.),
    lambda x: x.at[jnp.array([0, 1])].set(jnp.array([1., 2.])),
])
def test_unsupported_scatter_fails_closed(fn):
    with pytest.raises(TranslationError):
        certify(jax.make_jaxpr(fn)(jnp.ones(3)))


def test_dynamic_index_rejected():
    with pytest.raises(TranslationError):
        certify(jax.make_jaxpr(lambda x, i: x.at[i].set(9.))(jnp.ones(3), jnp.int32(0)))


@pytest.mark.lean
def test_wrong_scatter_coordinate_fails_certificate():
    source = certify(jax.make_jaxpr(lambda x: x.at[0].set(9.))(jnp.ones(2)))
    assert 'Tensor.scatterSet x0 (0, ())' in source
    source = source.replace('Tensor.scatterSet x0 (0, ())', 'Tensor.scatterSet x0 (1, ())', 1)
    result = lean(source, 'WrongStaticSet.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


@pytest.mark.parametrize('fn,s,u', [
    (scatter_updates.replace_middle, (4,), (2,)),
    (scatter_updates.replace_column, (2, 3), (2,)),
    (scatter_updates.replace_selected, (4,), (2,)),
    (scatter_updates.accumulate_selected, (4,), (3,)),
    (lambda x, u: x.at[1:3].add(u), (4,), (2,)),
    (lambda x, u: x.at[1].add(u), (4,), ()),
    (lambda x, u: x.at[-1].set(u), (4,), ()),
    (lambda x, u: x.at[1, :, 1:3].set(u), (2, 3, 4), (3, 2)),
    (lambda x, u: x.at[jnp.array([2, 0]), :].set(u), (3, 2), (2, 2)),
    (lambda x, u: x.at[jnp.array([-1, 0])].add(u), (4,), (2,)),
])
def test_scatter_plan_matches_jax(fn, s, u):
    x = np.arange(np.prod(s), dtype=np.float32).reshape(s)
    updates = np.arange(np.prod(u), dtype=np.float32).reshape(u) + 10
    closed = jax.make_jaxpr(fn)(jnp.asarray(x), jnp.asarray(updates))
    metadata = dict(zip(closed.jaxpr.constvars, map(np.asarray, closed.consts)))
    actual = x.copy()
    for eq in closed.jaxpr.eqns:
        static = integer_equation(eq, metadata)
        if static is not None:
            metadata[eq.outvars[0]] = static
        if eq.primitive.name in ('scatter', 'scatter-add'):
            for dest, src in scatter_plan(eq, metadata):
                if eq.primitive.name == 'scatter-add':
                    actual[dest] += updates[src]
                else:
                    actual[dest] = updates[src]
    np.testing.assert_array_equal(actual, fn(jnp.asarray(x), jnp.asarray(updates)))
    source = certify(closed)
    assert '_translation_correct' in source


@pytest.mark.parametrize('fn', [
    lambda x, u: x.at[jnp.array([1, 1])].set(u),
    lambda x, u: x.at[jnp.array([0, 8])].add(u),
    lambda x, u: x.at[jnp.array([0, 8])].set(u, mode='clip'),
])
def test_ambiguous_or_out_of_bounds_scatter_rejected(fn):
    with pytest.raises(TranslationError, match='overlap|in bounds'):
        certify(jax.make_jaxpr(fn)(jnp.ones(4), jnp.ones(2)))


@pytest.mark.lean
def test_wrong_scatter_combiner_fails_certificate():
    source = certify(jax.make_jaxpr(lambda x, u: x.at[1].add(u))(jnp.ones(4), jnp.float32(2)))
    source = source.replace('Tensor.scatterAdd', 'Tensor.scatterSet', 1)
    result = lean(source, 'WrongScatterCombiner.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


def test_scatter_rejects_overlapping_windows_even_with_unique_promise():
    dims = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(), scatter_dims_to_operand_dims=(0,))
    fn = lambda x, u: jax.lax.scatter(x, jnp.array([[0], [1]]), u, dims, unique_indices=True)
    with pytest.raises(TranslationError, match='overlap'):
        certify(jax.make_jaxpr(fn)(jnp.ones(4), jnp.ones((2, 2))))


def test_scatter_rejects_window_crossing_boundary():
    dims = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0,), inserted_window_dims=(), scatter_dims_to_operand_dims=(0,))
    fn = lambda x, u: jax.lax.scatter(x, jnp.array([3]), u, dims)
    with pytest.raises(TranslationError, match='in bounds'):
        certify(jax.make_jaxpr(fn)(jnp.ones(4), jnp.ones(2)))


def test_empty_scatter_and_expansion_limit():
    dims = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    fn = lambda x, u: jax.lax.scatter(x, jnp.empty((0, 1), dtype=jnp.int32), u, dims)
    source = certify(jax.make_jaxpr(fn)(jnp.ones(4), jnp.ones(0)))
    assert 'Tensor.scatterSet' not in source
    assert '.ret (.var .here)' in source
    fn = lambda x, u: x.at[:257].set(u)
    with pytest.raises(TranslationError, match='256-element'):
        certify(jax.make_jaxpr(fn)(jnp.ones(258), jnp.ones(257)))
