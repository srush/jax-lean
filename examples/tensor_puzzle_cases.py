"""Numbered coverage of srush/Tensor-Puzzles at ad5069f7ef48523f733d445a7f94e93427d64f2d."""
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from . import tensor_puzzles as puzzles


def array(*shape, dtype=jnp.float32):
    return jax.ShapeDtypeStruct(shape, dtype)


@dataclass(frozen=True)
class Puzzle:
    number: int
    name: str
    args: tuple
    size: int | None = None

    @property
    def module(self):
        return ''.join(word.title() for word in self.name.split('_'))

    def function(self, prefix):
        fn = getattr(puzzles, prefix + '_' + self.name)
        return partial(fn, n=self.size) if self.size is not None else fn


PUZZLES = (
    Puzzle(1, 'ones', (), 3),
    Puzzle(2, 'sum', (array(4),)),
    Puzzle(3, 'outer', (array(2), array(3))),
    Puzzle(4, 'diag', (array(3, 3),)),
    Puzzle(5, 'eye', (), 3),
    Puzzle(6, 'triu', (), 3),
    Puzzle(7, 'cumsum', (array(3),)),
    Puzzle(8, 'diff', (array(3),)),
    Puzzle(9, 'vstack', (array(3), array(3))),
    Puzzle(10, 'roll', (array(3),)),
    Puzzle(11, 'flip', (array(4),)),
    Puzzle(12, 'compress', (array(2, dtype=jnp.bool_), array(2))),
    Puzzle(13, 'pad_to', (array(3),), 5),
    Puzzle(14, 'sequence_mask', (array(2, 3), array(2, dtype=jnp.int32))),
    Puzzle(15, 'bincount', (array(3, dtype=jnp.int32),), 3),
    Puzzle(16, 'scatter_add', (array(3), array(3, dtype=jnp.int32)), 3),
    Puzzle(17, 'flatten', (array(2, 3),)),
    Puzzle(18, 'linspace', (array(), array()), 3),
    Puzzle(19, 'heaviside', (array(3), array(3))),
    Puzzle(20, 'repeat', (array(3),), 2),
    Puzzle(21, 'bucketize', (array(3), array(3))),
)

# Existing proofs and artifacts remain the source of truth for these four.
EXISTING = {2, 3, 11, 17}

# Keep the upstream inventory complete, but expose only checked examples.
DEFERRED = {12: "Compression: certificate checking needs a more efficient proof strategy."}
IMPLEMENTED = tuple(case for case in PUZZLES if case.number not in DEFERRED)
