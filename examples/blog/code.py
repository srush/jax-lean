import jax
import jax.numpy as jnp


def add_then_scale(a, b):
    total = a + b
    return total * 2.0


def eye(n):
    i = jnp.arange(n, dtype=jnp.float32)
    return jnp.where(i[:, None] == i[None, :], 1., 0.)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['add']:
        a = jnp.array([1., 2., 3.], dtype=jnp.float32)
        b = jnp.array([4., 5., 6.], dtype=jnp.float32)
        print(jax.make_jaxpr(add_then_scale)(a, b))
        raise SystemExit
    from examples.tensor_puzzles.code import puzzle_eye
    traced = jax.make_jaxpr(eye, static_argnums=(0,))(3)
    assert str(traced) == str(jax.make_jaxpr(puzzle_eye, static_argnums=(0,))(3))
    print(traced)
    print(eye(3))
