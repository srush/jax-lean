"""Fast dependency checks for the boundaries documented in ARCHITECTURE.md."""
import ast
from jaxlean_verso.notebook import load_manifest
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / 'JaxLean'


def imports(path):
    return re.findall(r'^import ((?:JaxLean(?:Examples)?|examples)(?:\.[A-Za-z0-9_]+)*)\s*$', path.read_text(), re.M)


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
    for path in [ROOT / 'JaxLean.lean', *LEAN.rglob('*.lean')]:
        assert not any(m.startswith(('JaxLeanExamples', 'examples.')) for m in imports(path)), path
    assert not (LEAN / 'Generated').exists()
    assert not (LEAN / 'Examples').exists()


def test_project_imports_and_notebook_modules_exist():
    for path in [ROOT / 'JaxLean.lean', ROOT / 'JaxLeanExamples.lean',
                 *LEAN.rglob('*.lean'), *(ROOT / 'examples').rglob('*.lean')]:
        for module in imports(path):
            assert (ROOT / (module.replace('.', '/') + '.lean')).is_file(), (path, module)
    config = load_manifest(ROOT / 'docs/examples.yaml')
    for page in config['pages']:
        for card in page['cards']:
            for item in card.get('python', []):
                assert (ROOT / item['file']).is_file()
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
    for name in ('Jaxpr',):
        source = (LEAN / 'Core' / f'{name}.lean').read_text()
        block = source.split('inductive Op ', 1)[1].split('noncomputable def Op.eval', 1)[0]
        constructors = re.findall(r'^  \| ([A-Za-z0-9_]+)', block, re.M)
        for constructor in constructors:
            assert f'`{constructor}`' in guide, (name, constructor)


def test_examples_keep_sources_proofs_and_generated_code_together():
    for generator in (ROOT / 'examples').glob('*/generate.py'):
        folder = generator.parent
        assert (folder / 'code.py').is_file(), folder
        assert list((folder / 'proofs').rglob('*.lean')), folder
        assert list((folder / 'generated').glob('*.lean')), folder
    assert not (ROOT / 'JaxLeanExamples').exists()
