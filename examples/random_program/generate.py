import jax
import jax.numpy as jnp
from jaxlean import transpile
from .code import sample_times_100, sample_scaled
from examples._support import main


def artifacts():
    for module, fn, inputs in (
        ("SampleTimes100", sample_times_100, (jax.random.key(0),)),
        ("SampleScaled", sample_scaled, (jax.random.key(0), jnp.float32(100))),
    ):
        yield module, transpile(fn, *inputs, namespace="JaxLean.Generated", random_model="uniform")


if __name__ == "__main__":
    main(__file__, artifacts)
