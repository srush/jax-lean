from pathlib import Path
import pytest
from jaxlean.docs import VersoPage, python_function, build


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
