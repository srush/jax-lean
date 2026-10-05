import jax
import jax.numpy as jnp
from jaxlean import certify_module
from . import code as scatter
from examples._support import main


def artifacts():
    def array(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float32)
    for module, name, fn, args in (
        ('ScatterMiddle', 'replace_middle', scatter.replace_middle, (array(4), array(2))),
        ('ScatterColumn', 'replace_column', scatter.replace_column, (array(2, 3), array(2))),
        ('ScatterSelected', 'replace_selected', scatter.replace_selected, (array(4), array(2))),
        ('ScatterAccumulate', 'accumulate_selected', scatter.accumulate_selected, (array(4), array(3))),
    ):
        namespace = 'JaxLean.ScatterJax'
        yield module, certify_module(jax.make_jaxpr(fn)(*args), name=name, namespace=namespace)


if __name__ == "__main__":
    main(__file__, artifacts)
