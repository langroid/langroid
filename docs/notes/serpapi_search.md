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

The integration returns up to `num_results` items from the first SerpApi Google
Search page's `organic_results`. A standard page currently contains at most
roughly 10 results; fetching more would require pagination with `start`, which
this integration intentionally does not perform. See
[`examples/basic/chat-search-serpapi.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search-serpapi.py)
for a complete example.
