"""Certify the array and imperative Jaxpr for each newly added tensor puzzle."""
import argparse
from pathlib import Path

import jax

from jaxlean import certify_module
from .tensor_puzzle_cases import EXISTING, IMPLEMENTED


def artifacts():
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
