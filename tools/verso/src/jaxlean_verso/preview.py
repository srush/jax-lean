"""Publish live-preview pages with reload detection already attached."""
import os
import re
import time

SCRIPT = """<script id="preview-reload" data-version="{version}">
(()=>{{
  const version = document.currentScript.dataset.version;
  let checking = false;
  async function check() {{
    if (checking) return;
    checking = true;
    try {{
      const response = await fetch('/preview-version.json', {{cache:'no-store'}});
      if (response.ok && (await response.text()).trim() !== version) location.reload();
    }} catch {{}} finally {{ checking = false; }}
  }}
  setInterval(check, 1000);
  window.addEventListener('pageshow', check);
  document.addEventListener('visibilitychange', () => {{ if (!document.hidden) check(); }});
}})();
</script>"""


def atomic_write(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def with_reload(document, version):
    # Replace both the original inline poller and the versioned poller.
    document = re.sub(
        r'<script\b[^>]*>[\s\S]*?</script>',
        lambda match: '' if 'preview-version.json' in match[0] else match[0],
        document,
    )
    return document.replace('</body>', SCRIPT.format(version=version) + '</body>')


def publish_reload(output, documents=None):
    version = str(time.time_ns())
    pages = {p.name: p.read_text() for p in output.glob('*.html')}
    pages.update(documents or {})
    for name, document in pages.items():
        atomic_write(output / name, with_reload(document, version))
    # Announce only after every page is ready.
    atomic_write(output / 'preview-version.json', version)
