"""Coverage and small differential checks against the original imperative specs."""
from jaxlean_verso.notebook import load_manifest
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pytest
from examples.tensor_puzzles.cases import PUZZLES, EXISTING, IMPLEMENTED, DEFERRED
from examples.tensor_puzzles.generate import artifacts
from examples.tensor_puzzles import code as puzzles
import tensor_puzzle_reference as reference

ROOT = Path(__file__).resolve().parents[1]


def test_complete_numbered_puzzle_inventory():
    assert [case.number for case in PUZZLES] == list(range(1, 22))
    assert {case.name + '_spec' for case in PUZZLES} == {
        name for name in vars(reference) if name.endswith('_spec')}
    pages = load_manifest(ROOT / 'docs/examples.yaml')['pages']
    page = next(p for p in pages if p['slug'] == 'puzzles')
    puzzle_titles = [card['title'] for card in page['cards']
                     if card.get('kind', 'puzzle') == 'puzzle' and card['title'].split('.', 1)[0].isdigit()]
    assert puzzle_titles == [f'{c.number}. {c.name}' for c in IMPLEMENTED]
    assert len(IMPLEMENTED) == 20
    assert set(DEFERRED) == {12}
    for case in IMPLEMENTED:
        if case.number not in EXISTING:
            assert (ROOT / f'examples/tensor_puzzles/proofs/Puzzles/{case.module}.lean').exists()


def test_puzzle_artifacts_current():
    for name, source in artifacts():
        assert (ROOT / 'examples/tensor_puzzles/generated' / (name + '.lean')).read_text() == source


@pytest.mark.parametrize('case', IMPLEMENTED, ids=lambda c: c.name)
def test_matches_original_spec(case):
    rng = np.random.default_rng(case.number)
    args = []
    for aval in case.args:
        if aval.dtype == jnp.bool_:
            data = np.array([True, False, True][:aval.shape[0]])
        elif aval.dtype == jnp.int32:
            data = rng.integers(0, 3, size=aval.shape, dtype=np.int32)
        else:
            data = rng.integers(-3, 4, size=aval.shape).astype(np.float32)
        args.append(data)
    if case.name == 'bucketize':
        args[1] = np.sort(args[1])
    result = np.asarray(case.function('puzzle')(*map(jnp.asarray, args)))
    expected = np.zeros_like(result.reshape(1) if result.ndim == 0 else result)
    spec_args = args + ([np.array([case.size])] if case.name == 'repeat' else [])
    getattr(reference, case.name + '_spec')(*spec_args, expected)
    np.testing.assert_allclose(result.reshape(expected.shape), expected, atol=1e-6)
    np.testing.assert_allclose(case.function('loop')(*map(jnp.asarray, args)), result, atol=1e-6)


def test_duplicate_links():
    values = jnp.array([2., 2., -1.])
    np.testing.assert_array_equal(puzzles.puzzle_bincount(jnp.array([1, 1, 1]), 3), [0, 3, 0])
    np.testing.assert_array_equal(puzzles.puzzle_scatter_add(values, jnp.array([1, 1, 1]), 3), [0, 3, 0])


def test_padding_linspace_and_bucket_edges():
    a = jnp.array([2., 4., 6.])
    for n in (0, 1, 3, 5):
        for name, args in [('pad_to', (a,)), ('linspace', (jnp.float32(-2), jnp.float32(4)))]:
            out = np.zeros(n, dtype=np.float32)
            getattr(reference, name + '_spec')(*args, out)
            np.testing.assert_allclose(getattr(puzzles, 'puzzle_' + name)(*args, n), out)
            np.testing.assert_allclose(getattr(puzzles, 'loop_' + name)(*args, n), out)
    v, boundaries = jnp.array([-1., 0., 1., 2., 3.]), jnp.array([0., 1., 1., 2.])
    out = np.zeros(5, dtype=np.float32)
    reference.bucketize_spec(v, boundaries, out)
    np.testing.assert_array_equal(puzzles.puzzle_bucketize(v, boundaries), out)
    np.testing.assert_array_equal(puzzles.loop_bucketize(v, boundaries), out)


@pytest.mark.lean
def test_wrong_iota_coordinate_fails_certificate():
    import jax
    from jaxlean import certify
    from test_random_translation import lean
    source = certify(jax.make_jaxpr(lambda: jnp.arange(3, dtype=jnp.float32))(), name='coordinates')
    assert '(i.1.val : R)' in source
    source = source.replace('(i.1.val : R)', '(i.1.rev.val : R)', 1)
    result = lean(source, 'WrongIotaCoordinate.lean')
    assert result.returncode != 0
    assert 'unsolved goals' in result.stdout


def test_mixed_static_updates_keep_runtime_indices_rejected():
    import jax
    from jaxlean import certify_module, TranslationError
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    mask = jax.ShapeDtypeStruct((), jnp.bool_)
    closed = jax.make_jaxpr(lambda x, m: x.at[1].set(jnp.where(m, x[0], 0.)))(x, mask)
    assert 'Jaxpr.Program' in certify_module(closed)
    with pytest.raises(TranslationError, match='static|real arguments'):
        certify_module(jax.make_jaxpr(lambda x, i: x.at[i].set(1.))(
            x, jax.ShapeDtypeStruct((), jnp.int32)))
