"""Fast dependency checks for the boundaries documented in ARCHITECTURE.md."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / 'JaxLean'


def imports(path):
    return re.findall(r'^import (JaxLean(?:\.[A-Za-z0-9_]+)*)\s*$', path.read_text(), re.M)


def test_lean_layers_depend_only_downward():
    allowed = {
        'Core': ('JaxLean.Core.',),
        'Verification': ('JaxLean.Core.', 'JaxLean.Verification.'),
        'Stdlib': ('JaxLean.Core.', 'JaxLean.Stdlib.'),
    }
    for layer, prefixes in allowed.items():
        for path in (LEAN / layer).glob('*.lean'):
            for module in imports(path):
                assert module.startswith(prefixes), (path, module)
                if layer == 'Stdlib':
                    assert module not in ('JaxLean.Core.Jaxpr', 'JaxLean.Core.TypedJaxpr'), (path, module)
    assert 'JaxLean.Examples' not in imports(ROOT / 'JaxLean.lean')
    assert not any(m.startswith('JaxLean.Examples.') for m in imports(ROOT / 'JaxLean.lean'))


def test_project_imports_and_notebook_modules_exist():
    import json
    for path in [ROOT / 'JaxLean.lean', *LEAN.rglob('*.lean')]:
        for module in imports(path):
            assert (ROOT / (module.replace('.', '/') + '.lean')).is_file(), (path, module)
    config = json.loads((ROOT / 'docs/examples.json').read_text())
    for page in config['pages']:
        for card in page['cards']:
            for item in card.get('lean', []):
                assert (ROOT / (item['module'].replace('.', '/') + '.lean')).is_file()


def test_importers_and_metadata_do_not_depend_on_emitter_or_certificate_assembly():
    files = [*(ROOT / 'python/jaxlean/importers').glob('*.py')]
    files += [ROOT / 'python/jaxlean' / name for name in ('jaxpr.py', 'layout.py', 'static_index.py')]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not set((node.module or '').split('.')) & {'translate', 'certify'}, (path, node.module)
                assert all(alias.name != 'Emitter' for alias in node.names), path


def test_primitive_index_covers_every_core_operation_constructor():
    guide = (ROOT / 'docs/primitive-index.md').read_text()
    for name in ('Jaxpr', 'TypedJaxpr'):
        source = (LEAN / 'Core' / f'{name}.lean').read_text()
        block = source.split('inductive Op ', 1)[1].split('noncomputable def Op.eval', 1)[0]
        constructors = re.findall(r'^  \| ([A-Za-z0-9_]+)', block, re.M)
        for constructor in constructors:
            assert f'`{constructor}`' in guide, (name, constructor)
