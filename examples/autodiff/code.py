import jax
import jax.numpy as jnp


def quadratic(x):
    return x * x + 3.0 * x


def jvp_tangent(x, v):
    _, tangent = jax.jvp(quadratic, (x,), (v,))
    return tangent


def gradient(x):
    return jax.grad(quadratic)(x)


if __name__ == '__main__':
    print(jax.make_jaxpr(gradient)(jax.ShapeDtypeStruct((), jnp.float32)))
