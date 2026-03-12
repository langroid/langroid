---

# **Using Seltz Search with Langroid**

---

## **1. Set Up Seltz**

1. **Access Seltz Platform**
   Go to [Seltz](https://seltz.ai/).

2. **Sign Up or Log In**
   Create an account or log in if you already have one.

3. **Get Your API Key**
   - Navigate to your dashboard
   - Copy your API key

4. **Set Environment Variable**
   Add the following variable to your `.env` file:
   ```env
   SELTZ_API_KEY=<your_api_key>
   ```

---

## **2. Install**

```bash
pip install langroid[seltz]
# or
uv pip install langroid[seltz]
```

---

## **3. Use Seltz Search with Langroid**

```python
import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.seltz_search_tool import SeltzSearchTool

# Configure the ChatAgent
config = ChatAgentConfig(
    name="search-agent",
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
)

# Create the agent and enable the Seltz search tool
agent = ChatAgent(config)
agent.enable_message(SeltzSearchTool)
```

---

## **4. Perform Web Searches**

Use the agent to perform web searches using Seltz.

```python
# Simple search query
response = agent.llm_response(
    "What are the latest developments in quantum computing?"
)
print(response)
```

---

## **5. Direct Tool Usage**

You can also use the tool directly without an agent:

```python
from langroid.agent.tools.seltz_search_tool import SeltzSearchTool

# Create a search request
search_request = SeltzSearchTool(
    query="Latest breakthroughs in fusion energy",
    num_results=3,
)

# Get search results
results = search_request.handle()
print(results)
```

---

## **6. Full Example**

See the complete working example at
[`examples/basic/chat-search-seltz.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search-seltz.py).

Run it with:
```bash
python3 examples/basic/chat-search-seltz.py
```

---
