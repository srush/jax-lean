import jax
import jax.numpy as jnp
from jaxlean import certify
from examples._support import main
from .code import add_then_scale


def artifacts():
    vector = jax.ShapeDtypeStruct((3,), jnp.float32)
    yield 'AddScale', certify(
        jax.make_jaxpr(add_then_scale)(vector, vector),
        name='add_then_scale', namespace='JaxLean.Blog', named_vars=True)


if __name__ == '__main__':
    main(__file__, artifacts)
