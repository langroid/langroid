---

# **Using Xquik Search with Langroid**

---

## **1. Set Up Xquik**

1. **Access Xquik**
   Go to [Xquik](https://xquik.com/).

2. **Get Your API Key**
   Create an API key from your Xquik dashboard.

3. **Set Environment Variable**
   Add the following variable to your `.env` file:
   ```env
   XQUIK_API_KEY=<your_api_key>
   ```

---

## **2. Use Xquik Search with Langroid**

```python
import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.xquik_search_tool import XquikSearchTool

# Configure the ChatAgent
config = ChatAgentConfig(
    name="x-search-agent",
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
)

# Create the agent and enable the Xquik search tool
agent = ChatAgent(config)
agent.enable_message(XquikSearchTool)
```

---

## **3. Search Public X Posts**

Use the agent to answer questions about public X posts. Queries can include
standard X search operators.

```python
response = agent.llm_response(
    'Search X for recent posts from xquikcom that mention "API".'
)
print(response)
```

---

## **4. Direct Tool Usage**

You can also use the tool directly without an agent:

```python
from langroid.agent.tools.xquik_search_tool import XquikSearchTool

search_request = XquikSearchTool(
    query='from:xquikcom "API"',
    num_results=3,
)

results = search_request.handle()
print(results)
```

---

## **5. Full Example**

See the complete working example at `examples/basic/xquik-search.py`.

Run it with:
```bash
python3 examples/basic/xquik-search.py
```

---
