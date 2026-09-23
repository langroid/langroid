# Web research with Baizhi MCP

The [example](https://github.com/langroid/langroid/blob/main/examples/mcp/baizhi-research.py)
connects Langroid to [Baizhi Agent Toolkit](https://baizhi.cloud/landing/agent-toolkit)
over Streamable HTTP. It discovers the server's schemas, enables only
`websearch_search`, `web_scrape`, and `web_extract`, and keeps the MCP connection
open for the task, closing it when the task exits.

## Setup

From a Langroid checkout with dependencies installed:

```bash
export BAIZHI_API_KEY='<your-baizhi-api-key>'
export OPENAI_API_KEY='<your-model-api-key>'
uv run examples/mcp/baizhi-research.py \
  --query='Read https://modelcontextprotocol.io/docs/getting-started/intro and summarize the protocol with source URLs.'
```

Obtain a Baizhi key from the service's account console. Keep keys out of source
control. Use `--model` to select another Langroid-supported model and configure
that model's credentials instead. `--turns=12` limits task turns; a turn can
involve multiple tool calls, so this is **not a spending or tool-call cap**.

## What the example does and does not enforce

- The service URL is fixed, the key is sent in the `Authorization` header, and
  the HTTP client does not follow redirects or use environment proxies. An empty
  key fails before connection.
- Schemas come from MCP discovery. If any of the three expected tools is absent
  or duplicated, the example stops. Other discovered tools are not enabled.
- The prompt requests small searches, no downloads, and source URLs. These are
  model instructions, not an enforced quota or a citation validator. Inspect
  results and verify important claims. Tool failures and task limits can prevent
  completion; increasing the turn limit can increase usage.
- The tool allowlist does not restrict the permissions of the service account.
  This example does not add a retry policy or a custom provider to Langroid.

Baizhi is a commercial hosted service. Tool calls can consume paid credits.
Queries, requested URLs, and extraction instructions go to Baizhi; tool results
also enter the configured model's context. Use public, nonsensitive inputs.
The [public integration repository](https://github.com/ct-jaryn/baizhi-agent-toolkit)
contains open-source client integrations; the hosted backend is not open source.
This community example does not imply a Langroid endorsement.

## Offline tests

```bash
uv run pytest tests/main/test_baizhi_mcp_example.py -q --nc --ns
```

The tests exercise Langroid's task and tool dispatch with an in-memory FastMCP
server and a simulated language model. A separate path exercises the real
Streamable HTTP transport with synthetic HTTP responses, including authentication
failure and redirects. Session cleanup is checked on success, model failure, and
cancellation during a tool call. The tests require no service or model keys and
make no production calls. They do not establish live service availability,
production authentication, billing, or model answer quality.
