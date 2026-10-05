import jax
import jax.numpy as jnp
from jaxlean import certify_module
from examples._support import main
from . import code as noether


def artifacts():
    vector = jax.ShapeDtypeStruct((4,), jnp.float32)
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    for name in ('advect', 'burgers'):
        yield 'Noether' + name.title(), certify_module(
            jax.make_jaxpr(getattr(noether, name))(vector, scalar),
            name=name, namespace='JaxLean.NoetherJax.' + name.title())


if __name__ == "__main__":
    main(__file__, artifacts)
