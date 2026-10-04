"""Generate certificates from the traced Jaxpr, without inspecting Python source."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import certify_module
from . import noether


def artifacts():
    vector = jax.ShapeDtypeStruct((4,), jnp.float32)
    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    for name in ('advect', 'burgers'):
        yield 'Noether' + name.title(), certify_module(
            jax.make_jaxpr(getattr(noether, name))(vector, scalar),
            name=name, namespace='JaxLean.NoetherJax.' + name.title())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / 'JaxLean/Generated'
    for name, source in artifacts():
        path = folder / (name + '.lean')
        if args.check:
            if not path.exists() or path.read_text() != source:
                raise SystemExit(f'stale generated file: {path}')
        else:
            path.write_text(source)
        print(path.name)


if __name__ == '__main__':
    main()
