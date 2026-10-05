import jax
import jax.numpy as jnp
from jaxlean import certify_module
from examples._support import main
from .code import transformer


def source():
    # Trace-specialized shapes; the mathematical library lemmas are generic.
    x = jax.ShapeDtypeStruct((3, 2), jnp.float32)
    w = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    jp = jax.make_jaxpr(transformer)(x, *([w] * 8))
    return certify_module(jp, name='transformer', namespace='JaxLean.TransformerJax')


def artifacts():
    yield "Transformer", source()


if __name__ == "__main__":
    main(__file__, artifacts)
