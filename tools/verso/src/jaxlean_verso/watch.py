"""Local editing preview with debounced builds and last-good-page reloads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from html.parser import HTMLParser
from .notebook import VersoPage, python_function, load_manifest
from .blog import render

RELOAD = '''<script>(()=>{let version;setInterval(async()=>{try{const r=await fetch('/preview-version.json',{cache:'no-store'});if(!r.ok)return;const v=await r.text();if(version!==undefined&&v!==version)location.reload();version=v;}catch{}},1000);})();</script>'''

class Outputs(HTMLParser):
    def __init__(self, source):
        super().__init__(); self.values=[]; self.current=None; self.feed(source)
    def handle_starttag(self, tag, attrs):
        if tag=='pre' and dict(attrs).get('class')=='output': self.current=[]
    def handle_data(self, value):
        if self.current is not None:self.current.append(value)
    def handle_endtag(self, tag):
        if tag=='pre' and self.current is not None:
            self.values.append(''.join(self.current));self.current=None

def snapshot(root):
    paths=[]
    for folder in ('docs','examples','JaxLean','tools/verso/src'):
        paths.extend(p for p in (root/folder).rglob('*') if p.suffix in ('.yaml','.yml','.json','.css','.svg','.py','.lean','.md') and '__pycache__' not in p.parts)
    paths.extend(root/p for p in ('lakefile.toml','lake-manifest.json','lean-toolchain','JaxLean.lean','JaxLeanExamples.lean'))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def publish_reload(output):
    for path in output.glob('*.html'):
        source=path.read_text()
        if RELOAD not in source:path.write_text(source.replace('</body>',RELOAD+'</body>'))
    (output/'preview-version.json').write_text(json.dumps(time.time_ns()))

def refresh(root, output):
    config=load_manifest(root/'docs/examples.yaml')
    page=next(p for p in config['pages'] if p.get('layout')=='blog')
    record=json.loads((output/'build.json').read_text())
    modules={i['module'] for c in page['cards'] for i in c['lean']}
    if not modules <= set(record['lean_modules']):raise ValueError('New Lean module requires a build')
    parsed={m:VersoPage((output/Path(*m.split('.'))/'index.html').read_text()) for m in modules}
    old=next(p for p in record['manifest']['pages'] if p.get('layout')=='blog')
    keys=[]
    for c in old['cards']:
        if 'blocks' in c:
            for b in c['blocks']:
                if b['type'] == 'trace_output': keys.append(b['example'])
                if b['type'] == 'eval_output': keys.append('eval')
        else:
            if c.get('trace'):keys.append('add' if c['trace']=='add' else 'eye')
            if c.get('evaluate'):keys.append('eval')
    values=Outputs((output/'blog.html').read_text()).values
    if len(keys)!=len(values):raise ValueError('Missing checked execution output')
    cached=dict(zip(keys,values))
    def execute(args, **kwargs):
        from types import SimpleNamespace
        key='eval' if args[0]=='lake' else 'add' if args[-1]=='add' else 'eye'
        return SimpleNamespace(stdout=cached[key])
    assets='\n'.join(parsed[sorted(modules)[0]].assets)
    document=render(page,parsed,assets,root,python_function,execute=execute)
    for name in ('blog.html','index.html'):
        temp=output/(name+'.tmp');temp.write_text(document);os.replace(temp,output/name)
    for name in ('blog.css','notebook.css','jax-pipeline.svg'):
        (output/name).write_bytes((root/'docs'/name).read_bytes())
    record['manifest']=config
    (output/'build.json').write_text(json.dumps(record,indent=2)+'\n')

def main():
    root=Path.cwd();output=root/'.lake/build/proof-notebook'
    seen=snapshot(root);publish_reload(output)
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
            result=subprocess.run([sys.executable,'-m','jaxlean_verso','build'],cwd=root)
            if result.returncode:
                needs_build=True
                print('Build failed; keeping the last successful preview.',flush=True);continue
            needs_build=False
        except Exception as error:
            print(f'Preview unchanged: {error}',flush=True);continue
        publish_reload(output)
        print('Preview updated: '+', '.join(sorted(changed)),flush=True)

if __name__=='__main__':main()
