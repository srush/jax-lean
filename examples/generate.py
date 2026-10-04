"""Run with: python -m examples.generate [--check]."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax

from jaxlean import translate
from .vector_norms import vector_norm, linear_clip, radial_clip, batch_radial_clip


def variance(x):
    """Population variance, written explicitly to avoid jnp.var's NaN branch."""
    centered = x - jnp.mean(x)
    return jnp.mean(centered * centered)


def prefix_sum(x):
    return lax.scan(lambda carry, a: (carry + a, carry + a), jnp.array(0.0), x)


def mlp(x, w1, b1, w2, b2):
    """A 2 → 3 → 2 forward pass; every row uses the same parameters."""
    hidden = jnp.maximum(x @ w1 + b1, 0.0)
    return hidden @ w2 + b2


def mlp_inputs(batch):
    # These values specify shapes/dtypes for tracing. Weights remain arguments.
    return (jnp.ones((batch, 2)), jnp.ones((2, 3)), jnp.ones(3),
            jnp.ones((3, 2)), jnp.ones(2))


def residual(x, w1, b1, w2, b2):
    """A different graph: 3 → 4 → 3, square activation, and a skip connection."""
    hidden = jnp.square(x @ w1 + b1)
    return x + (hidden @ w2 + b2)


def residual_inputs(batch):
    return (jnp.ones((batch, 3)), jnp.ones((3, 4)), jnp.ones(4),
            jnp.ones((4, 3)), jnp.ones(3))


EXAMPLES = {
    "VectorNorm": ("vector_norm", vector_norm, (jnp.ones(3),)),
    "LinearClip": ("linear_clip", linear_clip, (jnp.ones(3), jnp.ones((3, 2)), jnp.float32(1))),
    "RadialClip": ("radial_clip", radial_clip, (jnp.ones(3), jnp.float32(1))),
    "BatchRadialClip": ("batch_radial_clip", batch_radial_clip, (jnp.ones((4, 3)), jnp.float32(1))),
    "Residual5": ("residual5", residual, residual_inputs(5)),
    "Residual2": ("residual2", residual, residual_inputs(2)),
    "MLP4": ("mlp4", mlp, mlp_inputs(4)),
    "MLP3": ("mlp3", mlp, mlp_inputs(3)),
    "Mean": ("mean2", lambda x: jnp.mean(x), (jnp.ones(2),)),
    "Variance": ("variance2", variance, (jnp.ones(2),)),
    "Transpose": ("transpose23", lambda x: x.T, (jnp.ones((2, 3)),)),
    "TransposeTwice": ("transposeTwice", lambda x: x.T.T, (jnp.ones((2, 3)),)),
    "SumTranspose": ("sumTranspose", lambda x: jnp.sum(x.T), (jnp.ones((2, 3)),)),
    "Matmul": ("matmul", lambda x, y: x @ y, (jnp.ones((2, 3)), jnp.ones((3, 2)))),
    "Scan": ("prefixSum", prefix_sum, (jnp.ones(3),)),
    "Smooth": ("smooth", lambda x: jnp.exp(-x * x), (jnp.ones(3),)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in Lean has drifted")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for module, (name, fn, inputs) in EXAMPLES.items():
        closed = jax.make_jaxpr(fn)(*inputs)
        text = translate(closed, name=name, namespace="JaxLean.Generated",
                         readable=module in {"VectorNorm", "LinearClip", "RadialClip", "BatchRadialClip"})
        path = root / "JaxLean" / "Generated" / f"{module}.lean"
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f"stale generated file: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(f"{name}: {len(closed.jaxpr.eqns)} top-level Jaxpr equations → {path.name}")


if __name__ == "__main__":
    main()
