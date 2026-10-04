"""Generate randint estimators and certificates for kernels and tensor puzzles."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import certify_module, transpile
from . import tensor_puzzles as puzzles
from . import scatter_updates as scatter
from . import common_primitives as common
from . import randint_monte_carlo as mc


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
        ('ScatterMiddle', 'replace_middle', scatter.replace_middle, (array(4), array(2))),
        ('ScatterColumn', 'replace_column', scatter.replace_column, (array(2, 3), array(2))),
        ('ScatterSelected', 'replace_selected', scatter.replace_selected, (array(4), array(2))),
        ('ScatterAccumulate', 'accumulate_selected', scatter.accumulate_selected, (array(4), array(3))),
        ('LoopSum', 'loop_sum', puzzles.loop_sum, (array(4),)),
        ('LoopOuter', 'loop_outer', puzzles.loop_outer, (array(2), array(3))),
        ('LoopFlip', 'loop_flip', puzzles.loop_flip, (array(4),)),
        ('LoopFlatten', 'loop_flatten', puzzles.loop_flatten, (array(2, 3),)),
        ('PuzzleSum', 'sum', puzzles.puzzle_sum, (array(4),)),
        ('PuzzleFlip', 'flip', puzzles.puzzle_flip, (array(4),)),
        ('PuzzleOuterFlatten', 'outer_flatten', puzzles.outer_flatten, (array(2), array(3))),
        ('DieEstimate', 'die_estimate', mc.die_estimate, (array(16),)),
        ('GridEstimate', 'grid_estimate', mc.grid_estimate, (array(8),)),
    ):
        namespace = 'JaxLean.PuzzleJax' if module.startswith(('Puzzle', 'Loop')) else (f'JaxLean.CommonJax.{module[6:]}' if module.startswith('Common') else 'JaxLean.ScatterJax' if module.startswith('Scatter') else 'JaxLean.MCJax')
        yield module, certify_module(jax.make_jaxpr(fn)(*args), name=name, namespace=namespace)
    for module, fn in (('MCDie16', mc.mc_die_16), ('MCGridSquare8', mc.mc_grid_square_8)):
        yield module, transpile(fn, jax.random.key(0), namespace='JaxLean.MCJax',
                                random_model='uniform')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / 'JaxLean/Generated'
    for module, source in artifacts():
        path = folder / f'{module}.lean'
        if args.check:
            if not path.exists() or path.read_text() != source:
                raise SystemExit(f'stale generated file: {path}')
        else:
            path.write_text(source)
        print(path.name)


if __name__ == '__main__':
    main()
