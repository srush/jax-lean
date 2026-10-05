"""Small cross-check of the port's indexing convention against actual JAX."""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from examples.noether import code as noether
from examples.noether.generate import artifacts


def test_generated_noether_is_current():
    root = Path(__file__).resolve().parents[1] / 'examples/noether/generated'
    for name, source in artifacts():
        assert (root / (name + '.lean')).read_text() == source


def test_noether_periodic_orientation_and_equivariance():
    u = jnp.array([1., -2., 3., 5.])
    for fn, flux in ((noether.advect, lambda x: .25 * x),
                     (noether.burgers, lambda x: .25 * .5 * x**2)):
        result = fn(u, .25)
        f = flux(u)
        expected = np.array([u[i] - (f[i] - f[(i - 1) % 4]) for i in range(4)])
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result.sum(), u.sum())
        for shift in (-5, -1, 0, 1, 2, 7):
            np.testing.assert_array_equal(fn(jnp.roll(u, shift), .25), jnp.roll(result, shift))
