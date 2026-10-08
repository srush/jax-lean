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
        panel = '<div class="code-box">' + code(blocks[0].rstrip(), 'lean-source') + '</div>'
        if item.get('detail'):
            panel = '<details><summary>' + html.escape(item['name']) + '</summary>' + panel + '</details>'
        chunks.append(panel)
        chunks.append(text_slot(item, 'text_after', 'after ' + item['name']))
    return "".join(chunks)


def output_command(item, root):
    if item['runtime'] == 'python':
        return [sys.executable, '-m', item['module'], *map(str, item.get('args', []))]
    if item['runtime'] == 'lean':
        return ['lake', 'env', 'lean', str(root / item['file'])]
    raise ValueError(f"Unknown output runtime: {item['runtime']}")


def page_assets(page):
    return list(dict.fromkeys([
        *page.get('stylesheets', []),
        *(b['file'] for c in page['cards'] for b in c.get('blocks', []) if b['type'] == 'diagram'),
    ]))


def render(page, parsed, assets, root, extract, *, execute=subprocess.run):
    def lean(item):
        block = parsed[item['module']].blocks[item['name']]
        result = f'<div class="code-box">{block}</div>'
        if item.get('detail'): result=f'<details><summary>{html.escape(item["detail"])}</summary>{result}</details>'
        return result + text_slot(item, 'text_after', 'after ' + item['name'])
    title = html.escape(page['title'])
    sections=[]
    preamble=[]
    for card in page['cards']:
        chunks=[]
        before_contents = card.get('position') == 'before_contents'
        if not card.get('show_heading', True):
            chunks.append(f'<section id="{html.escape(card.get("id", ""))}">')
        elif card.get('puzzle'):
            chunks.append(f'<details class="puzzle"><summary>{html.escape(card["title"])}</summary>')
        else:
            heading = "h3" if card.get("subheading") else "h2"
            chunks.append(f'<section id="{card["id"]}"><{heading}>{html.escape(card["title"])}</{heading}>')
        for item in card['blocks']:
            kind = item['type']
            if kind == 'text':
                chunks.append(text_slot(item, 'text', item.get('marker', 'text')))
            elif kind == 'diagram':
                chunks.append(f'<figure><img src="{html.escape(item["file"])}" alt="{html.escape(item["alt"])}"></figure>')
            elif kind == 'datatype':
                chunks.append(render_datatypes([item], root))
            elif kind == 'math':
                chunks.append('<div class="equation">' + html.escape(item['text']) + '</div>')
            elif kind == 'python':
                source,_ = extract(root/item['file'],item['name'])
                chunks.append('<div class="code-box">' + code(source) + '</div>')
            elif kind == 'lean':
                chunks.append(lean(item))
            elif kind == 'code':
                source = (root / item['file']).read_text() if 'file' in item else item['text']
                if 'excerpt' in item:
                    start, end = item['excerpt']['start'], item['excerpt']['end']
                    if not start or source.count(start) != 1:
                        raise ValueError(f'Code excerpt start must match exactly once: {start!r}')
                    source = source[source.index(start):]
                    if not end or end not in source:
                        raise ValueError(f'Code excerpt end must occur after start: {end!r}')
                    source = source[:source.index(end)].rstrip()
                chunks.append('<div class="code-box">' + code(source, item['language']) + '</div>')
            elif kind == 'output':
                output = execute(output_command(item, root), cwd=root, check=True,
                                 capture_output=True, text=True).stdout
                chunks.append('<div class="code-box">' + code(output, 'output') + '</div>')
            else:
                raise ValueError(f'Unknown blog block type: {kind}')
        chunks.append('</details>' if card.get('puzzle') and card.get('show_heading', True) else '</section>')
        if before_contents:
            preamble.extend(chunks)
        else:
            sections.extend(chunks)
    toc=''.join(f'<li><a href="#{c["id"]}">{html.escape(c["title"])}</a></li>' for c in page['cards'] if c.get('toc', True) and c.get('position') != 'before_contents' and not c.get('puzzle') and not c.get('subheading'))
    styles = ''.join(f'<link rel="stylesheet" href="{html.escape(path)}">' for path in page.get('stylesheets', []))
    contents_title = html.escape(page.get('contents_title', 'Contents'))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>{assets}<link rel="stylesheet" href="tippy-border.css">{styles}</head><body><main><header><h1>{title}</h1>{''.join(preamble)}<nav aria-label="{contents_title}"><ol>{toc}</ol></nav></header>{''.join(sections)}</main></body></html>'''
