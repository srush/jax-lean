"""Pair displayed Python source with kernel-checked Verso Lean output.

Source inspection here is for documentation only; the transpiler still takes Jaxpr.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from .blog import highlight_python, render as render_blog, page_assets

MARKER = "jaxlean-verso-notebook-v1"


class VersoPage(HTMLParser):
    """Keep complete upstream code blocks, including nested tactic-state code."""
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.blocks = {}
        self.assets = []
        self.parts = None
        self.depth = 0
        self.kind = None
        self.names = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.parts is None:
            if tag == "code" and {"hl", "lean", "block"} <= set(attrs.get("class", "").split()):
                self.kind, self.parts, self.depth, self.names = "code", [], 0, []
            elif tag in ("script", "style") and (tag == "style" or attrs.get("src", "") in ("", "popper.js", "tippy.js")):
                self.kind, self.parts, self.depth = tag, [], 0
        if self.parts is not None:
            self.parts.append(self.get_starttag_text())
            if tag == self.kind:
                self.depth += 1
            binding = attrs.get("data-binding", "")
            if self.kind == "code" and "id" in attrs and binding.startswith("const-"):
                self.names.append(binding[6:])

    def handle_endtag(self, tag):
        if self.parts is not None:
            self.parts.append(f"</{tag}>")
            if tag == self.kind:
                self.depth -= 1
                if self.depth == 0:
                    value = "".join(self.parts)
                    if self.kind == "code":
                        for name in self.names:
                            if name in self.blocks:
                                raise ValueError(f"Ambiguous Lean declaration: {name}")
                            self.blocks[name] = value
                    elif self.kind == "style" or "window.onload" in value or 'src="popper.js"' in value or 'src="tippy.js"' in value:
                        self.assets.append(value)
                    self.parts = None

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")

    def handle_comment(self, data):
        self.handle_data(f"<!--{data}-->")


def python_function(path: Path, name: str) -> tuple[str, int]:
    source = path.read_text()
    matches = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one Python function {name} in {path}")
    node = matches[0]
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(source.splitlines()[start - 1:node.end_lineno]), start


def load_manifest(path):
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        config = yaml.safe_load(path.read_text())
    else:
        config = json.loads(path.read_text())
    for page in config.get('pages', []):
        for card in page['cards']:
            if 'blocks' in card:
                card['python'] = [b for b in card['blocks'] if b['type'] == 'python']
                card['lean'] = [b for b in card['blocks'] if b['type'] == 'lean']
                card['datatypes'] = [b for b in card['blocks'] if b['type'] == 'datatype']
    return config


def build(manifest: Path, output: Path, *, root: Path | None = None):
    root = Path.cwd() if root is None else root.resolve()
    config = load_manifest(manifest)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        marker = output / "build.json"
        if not marker.exists() or json.loads(marker.read_text()).get("generator") != MARKER:
            raise ValueError(f"Refusing to replace an unowned directory: {output}")
    # Execute explicit repository freshness checks, never code from the manifest.
    subprocess.run([sys.executable, "-m", "examples.generate", "--check"], cwd=root, check=True)
    modules = sorted({item["module"] for page in config["pages"] for card in page["cards"] for item in card["lean"]})
    subprocess.run(["lake", "build", "verso-html", *[m + ":literate" for m in modules]], cwd=root, check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="proof-notebook-", dir=output.parent) as tmp:
        staging = Path(tmp)
        inputs, site = staging / "input", staging / "site"
        for module in modules:
            relative = Path(*module.split(".")).with_suffix(".json")
            target = inputs / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / ".lake/build/literate" / relative, target)
        subprocess.run(["lake", "exe", "verso-html", str(inputs), str(site)], cwd=root, check=True)
        parsed = {m: VersoPage((site / Path(*m.split(".")) / "index.html").read_text()) for m in modules}
        assets = "\n".join(parsed[modules[0]].assets)
        # Tippy also opens on focus: make Lean hover tokens keyboard accessible.
        assets += """<script>document.addEventListener('DOMContentLoaded', () => {
          document.querySelectorAll('.hl.lean [data-verso-hover]').forEach(token => {
            token.tabIndex = 0;
          });
        });</script>"""
        hashes = {}
        execution_outputs = {}
        def execute(args, **kwargs):
            result = subprocess.run(args, **kwargs)
            execution_outputs[json.dumps(args)] = result.stdout
            return result
        for page in config["pages"]:
            if page.get("layout") == "blog":
                for card in page["cards"]:
                    for item in card["lean"]:
                        if item["name"] not in parsed[item["module"]].blocks:
                            raise ValueError(f"Missing Lean declaration: {item['name']}")
                        path = Path(*item["module"].split(".")).with_suffix(".lean")
                        hashes[str(path)] = hashlib.sha256((root / path).read_bytes()).hexdigest()
                    for item in card.get("datatypes", []):
                        hashes[item["file"]] = hashlib.sha256((root / item["file"]).read_bytes()).hexdigest()
                    for item in card["python"]:
                        hashes[item["file"]] = hashlib.sha256((root / item["file"]).read_bytes()).hexdigest()
                paths = page_assets(page) + [
                    b['file'] for c in page['cards'] for b in c['blocks']
                    if b['type'] in ('code', 'output') and 'file' in b]
                for path in paths:
                    hashes[path] = hashlib.sha256((root / path).read_bytes()).hexdigest()
                document = render_blog(page, parsed, assets, root, python_function, execute=execute)
                (site / f'{page["slug"]}.html').write_text(document)
                continue
            cards = []
            used = set()
            for card in page["cards"]:
                python = []
                for item in card["python"]:
                    path = root / item["file"]
                    source, line = python_function(path, item["name"])
                    hashes[item["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
                    python.append(f'<div class="source-label">{html.escape(item["file"])}:{line}</div><pre class="python"><code>{highlight_python(source)}</code></pre>')
                lean = []
                sections = {}
                for item in card["lean"]:
                    name, module = item["name"], item["module"]
                    if name in used:
                        raise ValueError(f"Duplicate declaration on page: {name}")
                    used.add(name)
                    block = parsed[module].blocks.get(name)
                    if block is None:
                        raise ValueError(f"Verso did not render {name} in {module}")
                    path = Path(*module.split(".")).with_suffix(".lean")
                    hashes[str(path)] = hashlib.sha256((root / path).read_bytes()).hexdigest()
                    link = "/".join(module.split(".")) + "/index.html"
                    panel = f'<div class="lean-declaration"><a class="source-label" href="{link}">{html.escape(str(path))} ↗</a>{block}</div>'
                    if item.get("detail"):
                        panel = f'<details><summary>{html.escape(item["detail"])}</summary>{panel}</details>'
                    lean.append(panel)
                    sections.setdefault(item.get("section", "Correctness proof"), []).append(block)
                if page.get("layout") == "code-only":
                    cells = []
                    for label in card.get("sections", ("Pseudocode", "JAX NumPy", "Correctness proof", "Certificate")):
                        python_items = [item for item in card["python"] if item.get("section", "JAX NumPy") == label]
                        if python_items:
                            content = "".join(
                                '<pre class="python"><code>' + highlight_python(python_function(root / item["file"], item["name"])[0]) + '</code></pre>'
                                for item in python_items
                            )
                        else:
                            content = "".join(sections.get(label, []))
                            if not content:
                                raise ValueError(f"Missing {label} in {card['title']}")
                        cells.append(f'<section class="notebook-cell"><h3>{html.escape(label)}</h3>{content}</section>')
                    cards.append(f'<article><h2>{html.escape(card["title"])}</h2>{"".join(cells)}</article>')
                    continue
                cards.append(f'<article><h2>{html.escape(card["title"])}</h2><p class="description">{html.escape(card["description"])}</p><div class="pair"><section class="panel"><h3>Python · JAX</h3>{"".join(python)}</section><section class="panel lean-panel"><h3>Lean · checked proof</h3>{"".join(lean)}</section></div></article>')
            nav = "".join(f'<a {"aria-current=page" if p["slug"] == page["slug"] else ""} href="{p["slug"]}.html">{html.escape(p["title"])}</a>' for p in config["pages"])
            document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(page['title'])} · JAX / Lean</title><link rel="icon" href="data:,">{assets}<link rel="stylesheet" href="tippy-border.css"><link rel="stylesheet" href="notebook.css"></head><body><header><div class="eyebrow">JAX / LEAN <span>PROOF NOTEBOOK</span></div><h1>Read the code. Explore the proof.</h1><p>Ordinary JAX functions beside their Lean specifications and certificates.</p><nav>{nav}</nav></header><main><aside><strong>Reading this page</strong> Hover over Lean names for types and documentation. Click a tactic’s underline to inspect its proof state. File links open the complete Lean module.<p><strong>Scope.</strong> These proofs describe the real-valued semantic model. Sampling uses ideal independent finite laws; equivalence to JAX’s PRNG and floating-point execution is not proved. Pairings are editorial; the expandable certificates state the checked connection.</p></aside>{''.join(cards)}</main><footer>Built from repository source with Verso · Lean 4.33.0 · <a href="build.json">Build record</a></footer></body></html>'''
            if page.get("layout") == "code-only":
                document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(page['title'])}</title><link rel="icon" href="data:,">{assets}<link rel="stylesheet" href="tippy-border.css"><link rel="stylesheet" href="notebook.css"></head><body class="code-only"><main>{''.join(cards)}</main></body></html>'''
            (site / f'{page["slug"]}.html').write_text(document)
        shutil.copy2(site / f'{config["pages"][0]["slug"]}.html', site / "index.html")
        shutil.copy2(root / "docs/notebook.css", site / "notebook.css")
        for page in config['pages']:
            for asset in page_assets(page):
                target = site / asset
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / asset, target)
        (site / "build.json").write_text(json.dumps({"generator": MARKER, "lean_modules": modules, "source_sha256": hashes, "execution_outputs": execution_outputs, "manifest": config}, indent=2) + "\n")
        if os.environ.get('JAXLEAN_LIVE_PREVIEW') == '1':
            from .preview import publish_reload
            publish_reload(site)
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(site), output)
    print(f"Built {output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "serve"])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="JaxLean repository directory")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.manifest = args.manifest or args.root / "docs/examples.yaml"
    args.output = args.output or args.root / ".lake/build/proof-notebook"
    if args.command == "build":
        build(args.manifest, args.output, root=args.root)
    else:
        if not (args.output / "build.json").exists():
            parser.error("Build the notebook first")
        print(f"Open http://localhost:{args.port}", flush=True)
        with ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(args.output.resolve()))) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
