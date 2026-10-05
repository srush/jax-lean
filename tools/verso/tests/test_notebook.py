from pathlib import Path
import pytest
from jaxlean_verso.notebook import VersoPage, python_function, build


def test_python_extracts_exact_function_with_decorator_without_execution(tmp_path):
    path = tmp_path / 'example.py'
    path.write_text('raise RuntimeError("must not execute")\n@jax.jit\ndef f(x):\n    return x < 2\n\ndef other():\n    pass\n')
    assert python_function(path, 'f') == ('@jax.jit\ndef f(x):\n    return x < 2', 2)
    with pytest.raises(ValueError, match='Expected one'):
        python_function(path, 'missing')


def test_verso_retains_nested_proof_states_entities_and_definition_only():
    block = '<code class="hl lean block"><span data-binding="const-N.f" id="f">f</span><span data-binding="const-N.reference">r</span><label><input type="checkbox"><code>goal &lt; &#123;</code></label></code>'
    page = VersoPage('<script src="popper.js"></script>' + block)
    assert page.blocks == {'N.f': block}
    assert page.assets == ['<script src="popper.js"></script>']


def test_duplicate_definition_rejected():
    block = '<code class="hl lean block"><span data-binding="const-N.f" id="f">f</span></code>'
    with pytest.raises(ValueError, match='Ambiguous'):
        VersoPage(block + block)


def test_build_does_not_replace_unowned_output(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"pages": []}')
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'keep.txt').write_text('keep')
    with pytest.raises(ValueError, match='unowned'):
        build(manifest, output)
    assert (output / 'keep.txt').read_text() == 'keep'


def test_cli_uses_explicit_repository_from_another_directory(tmp_path, monkeypatch):
    import sys
    from jaxlean_verso import notebook

    root = tmp_path / 'repository'
    root.mkdir()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    calls = []
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, 'argv', ['jaxlean-verso', 'build', '--root', str(root)])
    monkeypatch.setattr(notebook, 'build', lambda manifest, output, **kw: calls.append((manifest, output, kw)))
    notebook.main()
    assert calls == [(root / 'docs/examples.yaml', root / '.lake/build/proof-notebook', {'root': root})]


def test_notebook_import_needs_no_translation_or_jax(tmp_path):
    import os
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[1] / 'src'
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run(
        [sys.executable, '-S', '-c',
         'import sys; import jaxlean_verso.notebook; '
         'assert "jaxlean" not in sys.modules; assert "jax" not in sys.modules'],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_python_highlighting_preserves_source_and_escapes_html():
    from html.parser import HTMLParser
    from jaxlean_verso.blog import highlight_python

    class Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
        def handle_data(self, value):
            self.parts.append(value)

    source = '@jax.jit\ndef eye(n):\n    # <script> is text\n    s = "a < b & c"\n    return jnp.where(n < 3, 1., 0.)'
    rendered = highlight_python(source)
    parsed = Text()
    parsed.feed(rendered)
    assert ''.join(parsed.parts) == source
    assert '<script>' not in rendered
    assert 'class="py-keyword"' in rendered
    assert 'class="py-function"' in rendered
    assert 'class="py-number"' in rendered


def test_blog_renders_interleaved_blocks_in_manifest_order(tmp_path):
    from types import SimpleNamespace
    from jaxlean_verso.blog import render
    from jaxlean_verso.notebook import load_manifest

    source = tmp_path / 'sample.py'
    source.write_text('def sample(x):\n    return x + 1\n')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('''pages:
- layout: blog
  cards:
  - id: example
    title: Example
    blocks:
    - type: lean
      module: Sample
      name: proof
    - type: text
      text: Between snippets
    - type: python
      file: sample.py
      name: sample
''')
    page = load_manifest(manifest)['pages'][0]
    assert page['cards'][0]['lean'][0]['module'] == 'Sample'
    parsed = {'Sample': SimpleNamespace(blocks={'proof': '<code>checked_proof</code>'})}
    output = render(page, parsed, '', tmp_path, python_function)
    assert output.index('checked_proof') < output.index('Between snippets') < output.index('py-function')
    page['cards'][0]['blocks'].reverse()
    output = render(page, parsed, '', tmp_path, python_function)
    assert output.index('py-function') < output.index('Between snippets') < output.index('checked_proof')


def test_datatype_rendering_keeps_constructor_groups(tmp_path):
    from jaxlean_verso.blog import render_datatypes

    (tmp_path / 'Core.lean').write_text('''inductive Op where
  | add

  | select_n

abbrev Op.extra := Nat

def evaluator := 0
''')
    rendered = render_datatypes([{'file': 'Core.lean', 'name': 'Op'}], tmp_path)
    assert '| add' in rendered
    assert '| select_n' in rendered
    assert 'evaluator' not in rendered
    assert 'Op.extra' not in rendered


def test_blog_manifest_includes_unified_types_and_autodiff():
    from jaxlean_verso.notebook import load_manifest
    from jaxlean_verso.blog import render_datatypes

    root = Path(__file__).resolve().parents[3]
    pages = load_manifest(root / 'docs/examples.yaml')['pages']
    blog = next(page for page in pages if page['slug'] == 'blog')
    cards = {card.get('id'): card for card in blog['cards']}
    assert 'autodiff' in cards
    datatypes = cards['jax-ir']['datatypes']
    assert {'DType', 'Ty', 'Value', 'Var', 'Atom', 'Op', 'Args', 'Program'} <= {
        item['name'] for item in datatypes}
    rendered = render_datatypes(datatypes, root)
    assert '| select_n' in rendered
    assert '| call' in rendered
