import jax
import jax.numpy as jnp
from jaxlean import transpile
from .code import monte_carlo_4, monte_carlo_square_8, sample_square_plus_one
from examples._support import main


def artifacts():
    for module, fn, inputs in (
        ("MonteCarlo4", monte_carlo_4, (jax.random.key(0),)),
        ("MonteCarloSquare8", monte_carlo_square_8, (jax.random.key(0),)),
        ("SampleSquarePlusOne", sample_square_plus_one, (jax.random.key(0),)),
    ):
        yield module, transpile(fn, *inputs, namespace="JaxLean.Generated", random_model="uniform")


if __name__ == "__main__":
    main(__file__, artifacts)
