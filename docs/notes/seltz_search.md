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

## **2. Use Seltz Search with Langroid**

### **Installation**

Install langroid with the `seltz` extra:

```bash
pip install langroid[seltz]
# or
uv pip install langroid[seltz]
```

### **Code Example**

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

# Create the agent
agent = ChatAgent(config)

# Enable Seltz search tool
agent.enable_message(SeltzSearchTool)
```

---

## **3. Perform Web Searches**

Use the agent to perform web searches using Seltz.

```python
# Simple search query
response = agent.llm_response(
    "What are the latest developments in quantum computing?"
)
print(response)

# Search with specific number of results
response = agent.llm_response(
    "Find 5 recent news articles about artificial intelligence."
)
print(response)
```

---

## **4. Custom Search Requests**

You can also create a `SeltzSearchTool` instance directly:

```python
from langroid.agent.tools.seltz_search_tool import SeltzSearchTool

# Create a custom search request
search_request = SeltzSearchTool(
    query="Latest breakthroughs in fusion energy",
    num_results=3
)

# Get search results
results = search_request.handle()
print(results)
```

---

## **5. Full Example**

See [`examples/basic/chat-search-seltz.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-search-seltz.py)
for a complete chatbot example with Seltz search.

---
