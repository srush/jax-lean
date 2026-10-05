import jax
import jax.numpy as jnp
import numpy as np
from examples.autodiff.code import quadratic, jvp_tangent, gradient


def test_jvp_arbitrary_tangents_and_unit_gradient():
    for x in (-4., 0., 2.5):
        x = jnp.asarray(x, dtype=jnp.float32)
        for v in (-3., 0., 2.):
            v = jnp.asarray(v, dtype=jnp.float32)
            primal, tangent = jax.jvp(quadratic, (x,), (v,))
            np.testing.assert_array_equal(primal, x*x + 3*x)
            np.testing.assert_array_equal(tangent, (2*x + 3)*v)
            np.testing.assert_array_equal(jvp_tangent(x, v), tangent)
        np.testing.assert_array_equal(gradient(x), jax.grad(quadratic)(x))
