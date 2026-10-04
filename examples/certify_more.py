"""Generate randint estimators and certificates for kernels and tensor puzzles."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import certify_module, transpile
from . import tensor_puzzles as puzzles
from . import randint_monte_carlo as mc


def artifacts():
    def array(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float32)
    for module, name, fn, args in (
        ('PuzzleSum', 'sum', puzzles.puzzle_sum, (array(4),)),
        ('PuzzleFlip', 'flip', puzzles.puzzle_flip, (array(4),)),
        ('PuzzleOuterFlatten', 'outer_flatten', puzzles.outer_flatten, (array(2), array(3))),
        ('DieEstimate', 'die_estimate', mc.die_estimate, (array(16),)),
        ('GridEstimate', 'grid_estimate', mc.grid_estimate, (array(8),)),
    ):
        namespace = 'JaxLean.PuzzleJax' if module.startswith('Puzzle') else 'JaxLean.MCJax'
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
