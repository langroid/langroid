# Spider Cloud crawler

Use [Spider Cloud](https://spider.cloud/docs/api/) to load web pages as Markdown
through `URLLoader`. The integration uses Langroid's existing `requests`
dependency; no extra package, model, or vector database is required.
Spider is opt-in. `URLLoader` still uses Trafilatura by default.

## Configuration

Set `SPIDER_API_KEY` in your environment or project `.env` file:

```dotenv
SPIDER_API_KEY=your-spider-api-key
```

An explicit `SpiderConfig(api_key=...)` takes precedence over the environment.
Keep real credentials out of source files.

| Option | Default | Meaning |
| --- | --- | --- |
| `api_key` | `SPIDER_API_KEY` or empty | Required for nonempty loads |
| `mode` | `"scrape"` | `"scrape"` loads each supplied page; `"crawl"` follows links |
| `limit` | `1` | Positive integer page cap **per seed URL**, sent only in crawl mode |
| `timeout` | `60` | Positive finite timeout in seconds for each HTTP request |

The timeout is the `requests` connect/read timeout, not a deadline for the
entire batch or remote crawl. `limit=0` (unlimited crawling) is not supported.
Spider Cloud requests use your account's credits. Start with the default
single-page mode and increase the crawl limit deliberately.

## Load individual pages

```python
from langroid.parsing.url_loader import SpiderConfig, URLLoader

documents = URLLoader(
    urls=["https://example.com"],
    crawler_config=SpiderConfig(),
).load()

for document in documents:
    print(document.metadata.source)
    print(document.content)
```

## Crawl from a seed URL

```python
documents = URLLoader(
    urls=["https://example.com"],
    crawler_config=SpiderConfig(mode="crawl", limit=5, timeout=120),
).load()
```

Multiple seed URLs are processed sequentially, each with its own page cap.
The integration requests Markdown and synchronous JSON results. It does not
add background jobs, streaming, automatic retries, or another-service fallback.

## Results and failures

- Each successful, nonempty page becomes a `Document`. Markdown is preserved
  and `metadata.is_chunk` is false, ready for downstream chunking.
- `metadata.source` uses the returned page URL. In scrape mode a missing URL
  falls back to the supplied URL. In crawl mode a missing URL is logged and
  skipped, because a discovered page cannot safely be attributed to the seed.
- Direct `.pdf`, `.docx`, and `.doc` URLs use Langroid's existing document
  parser and `ParsingConfig`, including its download limits. Install the
  appropriate document parser extra when needed. These may return chunks;
  empty or failed direct documents are not sent to Spider as a fallback.
  Extensionless URLs are sent to Spider without a local HEAD request.
- Empty input or an empty response returns an empty list. Empty input does
  not require a key or make a request.
- Missing keys and invalid configuration raise an error before requests.
  HTTP/connection errors, invalid JSON, failed pages, and malformed results
  are logged and skipped. Successful pages from other results remain available.
  Consequently, an empty list can also mean that all requests failed; inspect
  the warnings. Logs omit response bodies, exception messages, and credentials.

See the runnable
[example](https://github.com/langroid/langroid/blob/main/examples/docqa/spider_loader.py).
The focused tests mock the HTTP boundary and do not use API credentials:

```bash
uv run pytest tests/main/test_spider_crawler.py -q
```
