import jax
import jax.numpy as jnp
from jaxlean import certify_module
from . import code as common
from examples._support import main


def artifacts():
    def array(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float32)
    for module, name, fn, args in (
        ('CommonMasked', 'masked_values', common.masked_values, (array(3), jax.ShapeDtypeStruct((3,), jnp.bool_))),
        ('CommonScores', 'batched_scores', common.batched_scores, (array(2, 3, 4), array(2, 5, 4))),
        ('CommonSoftmax', 'row_softmax', common.row_softmax, (array(2, 3),)),
        ('CommonEmbeddings', 'embeddings', common.embeddings, (array(5, 3), jax.ShapeDtypeStruct((2,), jnp.int32))),
        ('CommonShifted', 'shifted_embeddings', common.shifted_embeddings, (array(5, 3), jax.ShapeDtypeStruct((2,), jnp.int32))),
        ('CommonRearrange', 'rearrange', common.rearrange, (array(2, 1, 3), array(2, 2, 3))),
        ('CommonExtrema', 'extrema', common.extrema, (array(2, 3, 4),)),
    ):
        namespace = f'JaxLean.CommonJax.{module[6:]}'
        yield module, certify_module(jax.make_jaxpr(fn)(*args), name=name, namespace=namespace)


if __name__ == "__main__":
    main(__file__, artifacts)
