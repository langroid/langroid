# Markdown and HTML Parsing

Available since Langroid v0.66.0.

`DocumentType` (in `langroid.parsing.document_parser`) includes two values for
plain-text-like formats, in addition to `PDF`, `DOCX`, `DOC`, `TXT`, `XLSX`,
`XLS`, `PPTX`:

- `DocumentType.MD = "md"` — Markdown; detected from a `.md` path/URL
  extension, a `text/markdown` or `text/x-markdown` MIME type (bytes input),
  or an explicit `doc_type="md"`.
- `DocumentType.HTML = "html"` — HTML; detected from a `.html`/`.htm`
  path/URL extension, a `text/html` MIME type (bytes input), or an explicit
  `doc_type="html"`.

Markdown and HTML are *not* handled by `DocumentParser.create()` (there is no
page-oriented parser subclass for them); calling `create()` with an MD/HTML
source raises a `ValueError`. Instead, use the static method
`DocumentParser.chunks_from_path_or_bytes()`, which extracts the text and
chunks it according to the `Parser` config:

```python
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import Parser, ParsingConfig

parser = Parser(ParsingConfig())

# From a file path: type inferred from the extension
chunks = DocumentParser.chunks_from_path_or_bytes("notes/readme.md", parser)

# From a URL: fetched over HTTP(S), type inferred from the extension
chunks = DocumentParser.chunks_from_path_or_bytes(
    "https://example.com/readme.md", parser
)

# From bytes: pass doc_type explicitly (see limitations below)
with open("page.html", "rb") as f:
    raw = f.read()
chunks = DocumentParser.chunks_from_path_or_bytes(raw, parser, doc_type="html")
```

Each returned `Document` chunk carries the source path (or `"bytes"`) in its
metadata.

## What each format does

- **Markdown** (`DocumentType.MD`): content is kept as raw Markdown — no tag
  stripping — except that a leading YAML front-matter block
  (`--- ... ---`) is removed when it contains at least one `key: value`
  line. CRLF line endings are normalized to LF before front-matter
  detection. Leading `---` blocks that do *not* look like YAML (e.g. a
  thematic break, or prose between two horizontal rules) are preserved.
- **HTML** (`DocumentType.HTML`): tags are stripped with BeautifulSoup's
  `get_text(separator="\n")`, so text from adjacent elements is separated by
  newlines rather than concatenated.

## Behavior changes (vs. langroid < 0.66)

If you have existing ingestion pipelines, note that chunk contents — and
therefore embeddings, similarity scores, and retrieval results — can differ
after this change:

- **`.md` files** were previously mis-detected as plain text and passed
  through BeautifulSoup's `get_text()`, which could silently drop
  Markdown-embedded HTML and text resembling tags (e.g. `<placeholder>`).
  They now return the raw Markdown source, minus any YAML front-matter.
- **`.html` files** previously used `get_text()` with no separator, which
  concatenated text of adjacent elements (`<li>A</li><li>B</li>` → `AB`).
  Extraction now uses `get_text(separator="\n")`, producing one line per
  element.
- **`.txt` files and unrecognized plain text** are unchanged: still passed
  through BeautifulSoup `get_text()` as before.

## Limitations

- **Front-matter ambiguity**: a document that *opens* with a prose block
  between two `---` lines, where the prose looks like a YAML key
  (a single word followed by a colon), is indistinguishable from YAML
  front-matter and *will* be stripped. For example:

  ```markdown
  ---
  Note: important
  ---
  body text
  ```

  Here `Note: important` matches the `key: value` shape, so the block is
  removed. This is inherent Markdown/YAML ambiguity: rejecting such blocks
  would break legitimate single-key front-matter (`title: My Post`) used by
  Jekyll/Hugo. Blocks whose lines do not look like YAML keys (multi-word
  prose before the colon, bare URLs, plain text) are preserved.
- **Markdown bytes are usually MIME-detected as plain text**: `libmagic`
  reports most Markdown content as `text/plain`, so when passing raw bytes
  *without* `doc_type`, the content is typically ingested as `TXT` and YAML
  front-matter is **not** stripped. Pass `doc_type="md"` explicitly to get
  Markdown treatment for bytes input.
- **MIME detection fallback**: for bytes input, if `python-magic` (or the
  native `libmagic` library it wraps) is unavailable or broken, MIME
  detection is skipped and ingestion falls back to the plain-text (UTF-8
  decode) heuristic, so plain-text bytes still ingest as `TXT`; binary
  formats then raise `ValueError` instead of being detected.
