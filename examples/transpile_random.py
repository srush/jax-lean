"""Build step: python -m examples.transpile_random [--check].

The source program is ordinary JAX in random_program.py; its proof is separate
in RandomProgramProofs.lean. The ideal sampler specification is selected here.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxlean import transpile
from .random_program import sample_times_100, sample_scaled
from .monte_carlo import monte_carlo_4, monte_carlo_square_8, sample_square_plus_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1] / "JaxLean" / "Generated"
    for module, fn, inputs in (
        ("SampleTimes100", sample_times_100, (jax.random.key(0),)),
        ("SampleScaled", sample_scaled, (jax.random.key(0), jnp.float32(100))),
        ("MonteCarlo4", monte_carlo_4, (jax.random.key(0),)),
        ("MonteCarloSquare8", monte_carlo_square_8, (jax.random.key(0),)),
        ("SampleSquarePlusOne", sample_square_plus_one, (jax.random.key(0),)),
    ):
        code = transpile(fn, *inputs, namespace="JaxLean.Generated", random_model="uniform")
        path = folder / f"{module}.lean"
        if args.check:
            if not path.exists() or path.read_text() != code:
                raise SystemExit(f"stale generated file: {path}")
        else:
            path.write_text(code)
        print(f"{fn.__name__} → {path.name}")


if __name__ == "__main__":
    main()
