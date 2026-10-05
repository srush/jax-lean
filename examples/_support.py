"""Write or check artifacts next to an example's Python source."""
import argparse
from pathlib import Path


def write_artifacts(generator, artifacts, *, check=False):
    folder = Path(generator).resolve().parent / "generated"
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in artifacts():
        path = folder / (name + ".lean")
        if check:
            if not path.exists() or path.read_text() != source:
                raise SystemExit(f"stale generated file: {path}")
        else:
            path.write_text(source)
        print(path.relative_to(folder.parent.parent))


def main(generator, artifacts):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_artifacts(generator, artifacts, check=args.check)
