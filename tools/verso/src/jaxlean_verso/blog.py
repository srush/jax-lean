"""Article layout; prose is deliberately represented by author markers."""
import builtins
import html
import io
import keyword
import re
import subprocess
import sys
import tokenize


def highlight_python(source):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    def offset(pos): return offsets[pos[0] - 1] + pos[1]
    parts, cursor, previous = [], 0, ''
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.ENDMARKER, tokenize.ENCODING): continue
        start, end = offset(token.start), offset(token.end)
        if start >= end: continue
        parts.append(html.escape(source[cursor:start]))
        kind = {tokenize.STRING:'string', tokenize.NUMBER:'number', tokenize.COMMENT:'comment', tokenize.OP:'operator'}.get(token.type)
        if token.type == tokenize.NAME:
            kind = 'keyword' if keyword.iskeyword(token.string) else 'function' if previous in ('def','class') else 'builtin' if token.string in dir(builtins) else None
        value = html.escape(source[start:end])
        parts.append(f'<span class="py-{kind}">{value}</span>' if kind else value)
        cursor, previous = end, token.string
    parts.append(html.escape(source[cursor:]))
    return ''.join(parts)


def code(source, language='python'):
    value = highlight_python(source) if language == 'python' else html.escape(source)
    return f'<pre class="{language}"><code>{value}</code></pre>'


def markdown_text(value):
    from markdown_it import MarkdownIt
    return '<div class="prose">' + MarkdownIt('commonmark', {'html': False}).render(value) + '</div>'


def text_slot(owner, key, label):
    if key not in owner:
        return ''
    value = owner[key] or ''
    if value.strip():
        return markdown_text(value)
    return '<p class="text-marker">[TEXT: ' + html.escape(label) + ']</p>'


def render_datatypes(items, root):
    chunks = []
    for item in items:
        path = root / item['file']
        pattern = r'(?m)^(?:abbrev|inductive) ' + re.escape(item['name']) + r'(?=[\s:])[\s\S]*?(?=\n[^\s]|\Z)'
        blocks = re.findall(pattern, path.read_text())
        if len(blocks) != 1:
            raise ValueError(f"Expected one datatype {item['name']} in {path}")
        panel = '<div class="code-box"><span class="source-label">' + html.escape(item['file']) + '</span>' + code(blocks[0].rstrip(), 'lean-source') + '</div>'
        if item.get('detail'):
            panel = '<details><summary>' + html.escape(item['name']) + '</summary>' + panel + '</details>'
        chunks.append(panel)
        chunks.append(text_slot(item, 'text_after', 'after ' + item['name']))
    return "".join(chunks)


def render(page, parsed, assets, root, extract, *, execute=subprocess.run):
    def lean(item):
        block = parsed[item['module']].blocks[item['name']]
        link = item['module'].replace('.', '/') + '/index.html'
        result = f'<div class="code-box"><a class="source-label" href="{link}">{html.escape(item["section"] if "section" in item else "Lean")} ↗</a>{block}</div>'
        if item.get('detail'): result=f'<details><summary>{html.escape(item["detail"])}</summary>{result}</details>'
        return result + text_slot(item, 'text_after', 'after ' + item['name'])
    chunks=[]
    for card in page['cards']:
        if card.get('puzzle'):
            chunks.append(f'<details class="puzzle"><summary>{html.escape(card["title"])}</summary>')
        else:
            heading = "h3" if card.get("subheading") else "h2"
            chunks.append(f'<section id="{card["id"]}"><{heading}>{html.escape(card["title"])}</{heading}>')
        for item in card['blocks']:
            kind = item['type']
            if kind == 'text':
                chunks.append(text_slot(item, 'text', item.get('marker', 'text')))
            elif kind == 'diagram':
                chunks.append('<figure><img src="jax-pipeline.svg" alt="JAX to Jaxpr to StableHLO / XLA, branching to TPU and GPU"></figure>')
            elif kind == 'datatype':
                chunks.append(render_datatypes([item], root))
            elif kind == 'math':
                chunks.append('<div class="equation">' + html.escape(item['text']) + '</div>')
            elif kind == 'python':
                source,_ = extract(root/item['file'],item['name'])
                chunks.append('<div class="code-box"><span class="source-label">' + html.escape(item.get('section', 'Python · JAX')) + '</span>' + code(source) + '</div>')
            elif kind == 'lean':
                chunks.append(lean(item))
            elif kind == 'trace_code':
                source = 'print(jax.make_jaxpr(eye, static_argnums=(0,))(3))\nprint(eye(3))'
                if item['example'] == 'add':
                    source = 'a = jnp.array([1., 2., 3.], dtype=jnp.float32)\nb = jnp.array([4., 5., 6.], dtype=jnp.float32)\nprint(jax.make_jaxpr(add_then_scale)(a, b))'
                chunks.append('<div class="code-box">' + code(source) + '</div>')
            elif kind == 'trace_output':
                arguments = ['add'] if item['example'] == 'add' else []
                output = execute([sys.executable, '-m', 'examples.blog.code', *arguments], cwd=root, check=True, capture_output=True, text=True).stdout
                chunks.append('<div class="code-box">' + code(output, 'output') + '</div>')
            elif kind == 'eval_code':
                chunks.append('<div class="code-box">' + code((root/'examples/blog/proofs/Eye.lean').read_text(), 'lean-source') + '</div>')
            elif kind == 'eval_output':
                output = execute(['lake', 'env', 'lean', str(root/'examples/blog/proofs/Eye.lean')], cwd=root, check=True, capture_output=True, text=True).stdout
                chunks.append('<div class="code-box">' + code(output, 'output') + '</div>')
            else:
                raise ValueError(f'Unknown blog block type: {kind}')
        chunks.append('</details>' if card.get('puzzle') else '</section>')
    toc=''.join(f'<li><a href="#{c["id"]}">{html.escape(c["title"])}</a></li>' for c in page['cards'] if not c.get('puzzle') and not c.get('subheading'))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lean-Verified Jax</title>{assets}<link rel="stylesheet" href="tippy-border.css"><link rel="stylesheet" href="blog.css"></head><body><main><header><h1>Lean-Verified Jax</h1><a class="repo" href="https://github.com/srush/jax-lean">GitHub ↗</a><p class="text-marker">[TEXT: subtitle / author]</p><nav aria-label="Contents"><ol>{toc}</ol></nav></header>{''.join(chunks)}</main></body></html>'''
