"""Local editing preview with debounced builds and last-good-page reloads."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .notebook import VersoPage, python_function, load_manifest
from . import blog
from .preview import publish_reload


def snapshot(root):
    paths=[]
    for folder in ('docs','examples','JaxLean','tools/verso/src'):
        paths.extend(p for p in (root/folder).rglob('*') if p.suffix in ('.yaml','.yml','.json','.css','.svg','.py','.lean','.md') and '__pycache__' not in p.parts)
    paths.extend(root/p for p in ('lakefile.toml','lake-manifest.json','lean-toolchain','JaxLean.lean','JaxLeanExamples.lean'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def refresh(root, output):
    from types import SimpleNamespace
    config = load_manifest(root / 'docs/examples.yaml')
    record = json.loads((output / 'build.json').read_text())
    cached = record.get('execution_outputs', {})

    def execute(args, **kwargs):
        key = json.dumps(args)
        if key not in cached:
            raise ValueError('New execution output requires a build')
        return SimpleNamespace(stdout=cached[key])

    renderer = importlib.reload(blog)
    documents = {}
    for page in config['pages']:
        if page.get('layout') != 'blog':
            previous = next((p for p in record['manifest']['pages']
                             if p['slug'] == page['slug']), None)
            if previous != page:
                raise ValueError('Changed notebook page requires a build')
            continue
        modules = {i['module'] for c in page['cards'] for i in c['lean']}
        if not modules <= set(record['lean_modules']):
            raise ValueError('New Lean module requires a build')
        parsed = {m: VersoPage((output / Path(*m.split('.')) / 'index.html').read_text())
                  for m in modules}
        assets = '\n'.join(parsed[sorted(modules)[0]].assets) if modules else ''
        documents[page['slug'] + '.html'] = renderer.render(
            page, parsed, assets, root, python_function, execute=execute)
        for asset in renderer.page_assets(page):
            target = output / asset
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / asset).read_bytes())
    first = config['pages'][0]['slug'] + '.html'
    documents['index.html'] = documents[first] if first in documents else (output / first).read_text()
    record['manifest'] = config
    (output / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    publish_reload(output, documents)

def main():
    root=Path.cwd();output=root/'.lake/build/proof-notebook'
    seen=snapshot(root)
    try:
        refresh(root,output)
    except Exception as error:
        print(f'Initial refresh deferred: {error}',flush=True)
        publish_reload(output)
    print('Watching source files; prose/style edits reuse checked Lean output.',flush=True)
    needs_build=False
    while True:
        time.sleep(.5)
        current=snapshot(root)
        if current==seen:continue
        time.sleep(.7)
        current=snapshot(root)
        changed={p for p in current.keys()|seen.keys() if current.get(p)!=seen.get(p)}
        seen=current
        needs_build |= any(not p.startswith('docs/') for p in changed)
        try:
            if needs_build:raise ValueError('Code changed')
            refresh(root,output)
        except ValueError:
            result=subprocess.run([sys.executable,'-m','jaxlean_verso','build'],cwd=root,
                                  env={**os.environ, 'JAXLEAN_LIVE_PREVIEW': '1'})
            if result.returncode:
                needs_build=True
                print('Build failed; keeping the last successful preview.',flush=True);continue
            needs_build=False
        except Exception as error:
            print(f'Preview unchanged: {error}',flush=True);continue
        print('Preview updated: '+', '.join(sorted(changed)),flush=True)

if __name__=='__main__':main()
