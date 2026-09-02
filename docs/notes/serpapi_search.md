# Using SerpApi Search with Langroid

SerpApi provides Google search results without requiring an additional Python
dependency. Create an API key at [SerpApi](https://serpapi.com/) and add it to
your `.env` file:

```env
SERPAPI_API_KEY=<your_api_key>
```

Enable the tool on an agent:

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.serpapi_search_tool import SerpApiSearchTool

agent = ChatAgent(ChatAgentConfig(name="search-agent"))
agent.enable_message(SerpApiSearchTool)
```

The integration returns up to `num_results` items from a single SerpApi Google
Search page's `organic_results`: `num_results` is passed through as SerpApi's
`num` parameter, and no further pages are fetched via `start`, so a very large
`num_results` may yield fewer results than requested. Organic results without a
`link` are skipped, since there would be no page to fetch content from. See
[`examples/basic/chat-search-serpapi.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search-serpapi.py)
for a complete example.
