"""Regenerate or check every self-contained example: python -m examples.generate."""
import argparse
from importlib import import_module
from pathlib import Path
from ._support import write_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path in sorted(Path(__file__).parent.glob("*/generate.py")):
        module = import_module(f"examples.{path.parent.name}.generate")
        write_artifacts(module.__file__, module.artifacts, check=args.check)


if __name__ == "__main__":
    main()
