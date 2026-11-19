# Using Firecrawl Search with Langroid

Firecrawl searches the web and returns a query-relevant snippet for each
result, and can also return each result's full page as markdown in the same
call. Create an API key at
[Firecrawl](https://www.firecrawl.dev/app/api-keys?utm_source=langroid&utm_medium=integration)
and add it to your `.env` file:

```env
FIRECRAWL_API_KEY=<your_api_key>
```

Install the `firecrawl` extra:

```bash
pip install "langroid[firecrawl]"
```

Enable the tool on an agent:

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.firecrawl_search_tool import FirecrawlSearchTool

agent = ChatAgent(ChatAgentConfig(name="search-agent"))
agent.enable_message(FirecrawlSearchTool)
```

The integration returns up to `num_results` web results (at most 100). The
snippet Firecrawl returns for each result fills its summary and content, so no
extra request is made to fetch the page. Calling
`firecrawl_search(query, num_results, scrape=True)` directly also scrapes each
result and uses its full page as markdown (JS-rendered pages included); each
scraped page costs extra Firecrawl credits and adds time. A result without
content falls back to fetching its link. See
[`examples/basic/chat-search-firecrawl.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search-firecrawl.py)
for a complete example.

The same extra and key also power `FirecrawlCrawler` for loading known URLs;
see the [URLLoader note](url_loader.md).
