"""Trace an ordinary JAX function and emit a checkable translation certificate."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import certify


def mean(x):
    return jnp.mean(x)


def relu_layer(x, weights, bias):
    return jnp.maximum(x @ weights + bias, 0.0)


def gram(x):
    return x.T @ x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for module, name, fn, inputs in (
        ('CertifiedMean', 'certified_mean', mean, (jnp.ones(2),)),
        ('CertifiedLayer', 'certified_layer', relu_layer,
         (jnp.ones((3, 2)), jnp.ones((2, 4)), jnp.ones(4))),
        ('CertifiedGram', 'certified_gram', gram, (jnp.ones((3, 2)),)),
    ):
        source = certify(jax.make_jaxpr(fn)(*inputs), name=name, namespace='JaxLean.Generated')
        path = Path(__file__).resolve().parents[1] / 'JaxLean/Generated' / f'{module}.lean'
        if args.check:
            if not path.exists() or path.read_text() != source:
                raise SystemExit(f'stale generated file: {path}')
        else:
            path.write_text(source)
        print(path.name)


if __name__ == '__main__':
    main()
