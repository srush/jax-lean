import jax
import jax.numpy as jnp
from jaxlean import translate
from .code import mlp, mlp_inputs, residual, residual_inputs
from examples._support import main

EXAMPLES = {
    'Residual5': ("residual5", residual, residual_inputs(5)),
    'Residual2': ("residual2", residual, residual_inputs(2)),
    'MLP4': ("mlp4", mlp, mlp_inputs(4)),
    'MLP3': ("mlp3", mlp, mlp_inputs(3)),
}


def artifacts():
    for module, (name, fn, inputs) in EXAMPLES.items():
        yield module, translate(jax.make_jaxpr(fn)(*inputs), name=name,
                                namespace="JaxLean.Generated", readable=False)


if __name__ == "__main__":
    main(__file__, artifacts)
