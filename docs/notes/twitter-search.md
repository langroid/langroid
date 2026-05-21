# Twitter Search via Xquik MCP

Xquik exposes a remote MCP server for X post search, user lookup, monitoring,
webhooks, and write actions. Langroid can use that server directly through its
MCP support, so no custom Langroid tool is required.

Xquik is a third-party service. Review its terms, privacy policy, and API key
handling before sending keys or search queries to the remote MCP endpoint.

## Setup

Create an API key in the Xquik dashboard, then export it before running the
example:

```bash
export XQUIK_API_KEY=<your-xquik-api-key>
```

The example connects to the remote Streamable HTTP MCP endpoint:

```python
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://xquik.com/mcp",
    headers={"x-api-key": api_key},
)
```

## Run the Example

```bash
uv run examples/mcp/xquik-twitter-search.py -m gpt-4.1-mini
```

You can pass a query and result limit:

```bash
uv run examples/mcp/xquik-twitter-search.py \
  --query 'from:xquikcom "API"' \
  --limit 5
```

The default query uses X advanced search syntax.

The script loads the Xquik MCP tools with `get_tools_async`, enables them on a
`ChatAgent`, and asks the agent to search X for matching posts.

See the full example in `examples/mcp/xquik-twitter-search.py`.
