---

# **Using Tavily Search with Langroid**

---

## **1. Set Up Tavily**

1. **Access Tavily Platform**  
   Go to the [Tavily Platform](https://tavily.com/).
   
2. **Sign Up or Log In**  
   Create an account or log in if you already have one.

3. **Get Your API Key**  
   - Navigate to your dashboard
   - Copy your API key

4. **Set Environment Variable**  
   Add the following variable to your `.env` file:
   ```env
   TAVILY_API_KEY=<your_api_key>

---

## **2. Use Tavily Search with Langroid**

### **Installation**

```bash
uv add tavily-python
# or
pip install tavily-python
```
### **Code Example**

```python
import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.tavily_search_tool import TavilySearchTool

# Configure the ChatAgent
config = ChatAgentConfig(
    name="search-agent",
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
    use_tools=True
)

# Create the agent
agent = ChatAgent(config)

# Enable Tavily search tool
agent.enable_message(TavilySearchTool)

```
---

## **3. Perform Web Searches**

Use the agent to perform web searches using Tavily's AI-powered search.

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

You can also customize the search behavior by creating a TavilySearchTool instance directly:

```python
from langroid.agent.tools.tavily_search_tool import TavilySearchTool

# Create a custom search request
search_request = TavilySearchTool(
    query="Latest breakthroughs in fusion energy",
    num_results=3
)

# Get search results
results = search_request.handle()
print(results)
```

---