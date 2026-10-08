from jaxlean_verso import preview


def test_pages_have_reload_before_new_version_is_announced(tmp_path, monkeypatch):
    (tmp_path / 'index.html').write_text('<body>old</body>')
    (tmp_path / 'other.html').write_text('<body>other</body>')
    writes = []
    replace = preview.os.replace

    def observe(source, target):
        writes.append(target.name)
        if target.name.endswith('.html'):
            assert 'id="preview-reload"' in source.read_text()
        else:
            version = source.read_text()
            for page in tmp_path.glob('*.html'):
                assert f'data-version="{version}"' in page.read_text()
        replace(source, target)

    monkeypatch.setattr(preview.os, 'replace', observe)
    preview.publish_reload(tmp_path, {'index.html': '<body>new</body>'})
    assert writes[-1] == 'preview-version.json'
    assert '>new<script' in (tmp_path / 'index.html').read_text()


def test_republishing_replaces_old_poller_and_preserves_other_scripts():
    source = """<body><script src="math.js"></script>
<script>(()=>{fetch('/preview-version.json')})();</script></body>"""
    updated = preview.with_reload(preview.with_reload(source, '100'), '200')
    assert updated.count('id="preview-reload"') == 1
    assert 'data-version="200"' in updated
    assert 'data-version="100"' not in updated
    assert '<script src="math.js"></script>' in updated
