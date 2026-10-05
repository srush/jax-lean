import jax
import jax.numpy as jnp
from jaxlean import certify
from examples._support import main
from .code import add_vectors


def artifacts():
    vector = jax.ShapeDtypeStruct((3,), jnp.float32)
    yield 'AddVectors', certify(
        jax.make_jaxpr(add_vectors)(vector, vector),
        name='add_vectors', namespace='JaxLean.Blog')


if __name__ == '__main__':
    main(__file__, artifacts)
