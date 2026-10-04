"""Run: python -m examples.sampling [--check]. Then lake build.

The claims below are exact rational numbers. The exporter writes proofs about
the actual transpiled operations, and Lean rejects incorrect claims.
"""
import argparse
from fractions import Fraction
from pathlib import Path

import jax

from jaxlean import Discrete


def stages():
    x = Discrete([-1, 2], weights=[1, 3])
    y = x.map(lambda v: 2 * v + 1)
    z = y.map(lambda v: v * v)
    average = z.independent_map2(z, lambda a, b: (a + b) / 2)
    reused = z.map(lambda v: v + v)
    return {
        "Draw": ("draw", x, Fraction(5, 4), Fraction(27, 16)),
        "AffineDraw": ("affineDraw", y, Fraction(7, 2), Fraction(27, 4)),
        "SquaredDraw": ("squaredDraw", z, 19, 108),
        "AverageDraw": ("averageDraw", average, 19, 54),
        "ReusedDraw": ("reusedDraw", reused, 38, 432),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / "JaxLean" / "Generated"
    for module, (name, rv, mean, variance) in stages().items():
        source = rv.to_lean(name=name, mean=mean, variance=variance)
        path = folder / f"{module}.lean"
        if args.check:
            if not path.exists() or path.read_text() != source:
                raise SystemExit(f"stale generated file: {path}")
        else:
            path.write_text(source)
        values, probabilities = rv.enumerate()
        print(f"{name}: values={values.tolist()}, probabilities={list(map(str, probabilities))}")
        print(f"  exact claims: mean={mean}, variance={variance}")
    if not args.check:
        average = stages()["AverageDraw"][1]
        print("JAX samples:", average.sample(jax.random.key(0), (8,)).tolist())


if __name__ == "__main__":
    main()
