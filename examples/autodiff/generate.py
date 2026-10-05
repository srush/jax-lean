import jax
import jax.numpy as jnp
from jaxlean import certify
from examples._support import main
from .code import quadratic, jvp_tangent, gradient


def artifacts():
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    for module, name, fn, args in (
        ('Quadratic', 'quadratic', quadratic, (scalar,)),
        ('JvpTangent', 'jvp_tangent', jvp_tangent, (scalar, scalar)),
        ('Gradient', 'gradient', gradient, (scalar,)),
    ):
        yield module, certify(jax.make_jaxpr(fn)(*args), name=name,
                              namespace='JaxLean.Autodiff')


if __name__ == '__main__':
    main(__file__, artifacts)
