import jax
import jax.numpy as jnp
from jaxlean import certify
from .code import mean, relu_layer, gram
from examples._support import main


def artifacts():
    for module, name, fn, inputs in (
        ('CertifiedMean', 'certified_mean', mean, (jnp.ones(2),)),
        ('CertifiedLayer', 'certified_layer', relu_layer,
         (jnp.ones((3, 2)), jnp.ones((2, 4)), jnp.ones(4))),
        ('CertifiedGram', 'certified_gram', gram, (jnp.ones((3, 2)),)),
    ):
        yield module, certify(jax.make_jaxpr(fn)(*inputs), name=name, namespace="JaxLean.Generated")


if __name__ == "__main__":
    main(__file__, artifacts)
