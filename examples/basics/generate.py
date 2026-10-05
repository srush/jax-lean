import jax
import jax.numpy as jnp
from jaxlean import translate
from .code import variance, prefix_sum
from examples._support import main

EXAMPLES = {
    'Mean': ("mean2", lambda x: jnp.mean(x), (jnp.ones(2),)),
    'Variance': ("variance2", variance, (jnp.ones(2),)),
    'Transpose': ("transpose23", lambda x: x.T, (jnp.ones((2, 3)),)),
    'TransposeTwice': ("transposeTwice", lambda x: x.T.T, (jnp.ones((2, 3)),)),
    'SumTranspose': ("sumTranspose", lambda x: jnp.sum(x.T), (jnp.ones((2, 3)),)),
    'Matmul': ("matmul", lambda x, y: x @ y, (jnp.ones((2, 3)), jnp.ones((3, 2)))),
    'Scan': ("prefixSum", prefix_sum, (jnp.ones(3),)),
    'Smooth': ("smooth", lambda x: jnp.exp(-x * x), (jnp.ones(3),)),
}


def artifacts():
    for module, (name, fn, inputs) in EXAMPLES.items():
        yield module, translate(jax.make_jaxpr(fn)(*inputs), name=name,
                                namespace="JaxLean.Generated", readable=False)


if __name__ == "__main__":
    main(__file__, artifacts)
