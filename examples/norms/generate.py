import jax
import jax.numpy as jnp
from jaxlean import translate
from .code import vector_norm, linear_clip, norm_clip, batch_norm_clip
from examples._support import main

EXAMPLES = {
    'VectorNorm': ("vector_norm", vector_norm, (jnp.ones(3),)),
    'LinearClip': ("linear_clip", linear_clip, (jnp.ones(3), jnp.ones((3, 2)), jnp.float32(1))),
    'NormClip': ("norm_clip", norm_clip, (jnp.ones(3), jnp.float32(1))),
    'BatchNormClip': ("batch_norm_clip", batch_norm_clip, (jnp.ones((4, 3)), jnp.float32(1))),
}


def artifacts():
    for module, (name, fn, inputs) in EXAMPLES.items():
        yield module, translate(jax.make_jaxpr(fn)(*inputs), name=name,
                                namespace="JaxLean.Generated", readable=True)


if __name__ == "__main__":
    main(__file__, artifacts)
