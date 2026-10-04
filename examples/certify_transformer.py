"""Generate a compositional certificate from the transformer's Jaxpr only."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import certify_module
from .transformer import transformer


def source():
    # Trace-specialized shapes; the mathematical library lemmas are generic.
    x = jax.ShapeDtypeStruct((3, 2), jnp.float32)
    w = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    jp = jax.make_jaxpr(transformer)(x, *([w] * 8))
    return certify_module(jp, name='transformer', namespace='JaxLean.TransformerJax')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'JaxLean/Generated/Transformer.lean'
    generated = source()
    if args.check:
        if not path.exists() or path.read_text() != generated:
            raise SystemExit(f'stale generated file: {path}')
    else:
        path.write_text(generated)
    print(path.name)


if __name__ == '__main__':
    main()
