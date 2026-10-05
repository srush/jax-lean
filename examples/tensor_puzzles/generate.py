import jax
import jax.numpy as jnp
from jaxlean import certify_module
from . import code as puzzles
from .cases import EXISTING, IMPLEMENTED
from examples._support import main


def artifacts():
    def array(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float32)
    for module, name, fn, args in (
        ('LoopSum', 'loop_sum', puzzles.loop_sum, (array(4),)),
        ('LoopOuter', 'loop_outer', puzzles.loop_outer, (array(2), array(3))),
        ('LoopFlip', 'loop_flip', puzzles.loop_flip, (array(4),)),
        ('LoopFlatten', 'loop_flatten', puzzles.loop_flatten, (array(2, 3),)),
        ('PuzzleSum', 'sum', puzzles.puzzle_sum, (array(4),)),
        ('PuzzleFlip', 'flip', puzzles.puzzle_flip, (array(4),)),
        ('PuzzleOuterFlatten', 'outer_flatten', puzzles.outer_flatten, (array(2), array(3))),
    ):
        namespace = 'JaxLean.PuzzleJax'
        yield module, certify_module(jax.make_jaxpr(fn)(*args), name=name, namespace=namespace)
    yield from puzzle_artifacts()


def puzzle_artifacts():
    for case in IMPLEMENTED:
        if case.number in EXISTING:
            continue
        sources = []
        for prefix, layer in [('puzzle', 'Array'), ('loop', 'Loop')]:
            closed = jax.make_jaxpr(case.function(prefix))(*case.args)
            sources.append(certify_module(
                closed, name=case.name,
                namespace=f'JaxLean.Puzzles.{case.module}.{layer}',
            ))
        lines = '\n'.join(sources).splitlines()
        imports = list(dict.fromkeys(line for line in lines if line.startswith('import ')))
        options = ['set_option maxRecDepth 4096', 'set_option maxHeartbeats 2000000']
        body = [line for line in lines if not line.startswith('import ')]
        yield 'Puzzles' + case.module, '\n'.join(imports + options + body) + '\n'


if __name__ == "__main__":
    main(__file__, artifacts)
