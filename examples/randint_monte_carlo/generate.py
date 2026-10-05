import jax
import jax.numpy as jnp
from jaxlean import certify_module, transpile
from . import code as mc
from examples._support import main


def artifacts():
    def array(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float32)
    for module, name, fn, args in (
        ('DieEstimate', 'die_estimate', mc.die_estimate, (array(16),)),
        ('GridEstimate', 'grid_estimate', mc.grid_estimate, (array(8),)),
    ):
        namespace = 'JaxLean.MCJax'
        yield module, certify_module(jax.make_jaxpr(fn)(*args), name=name, namespace=namespace)
    for module, fn in (("MCDie16", mc.mc_die_16), ("MCGridSquare8", mc.mc_grid_square_8)):
        yield module, transpile(fn, jax.random.key(0), namespace="JaxLean.MCJax", random_model="uniform")


if __name__ == "__main__":
    main(__file__, artifacts)
