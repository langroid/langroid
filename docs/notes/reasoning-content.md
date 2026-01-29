# Stream and capture reasoning content in addition to final answer, from Reasoning LLMs

As of v0.35.0, when using certain Reasoning LLM APIs (e.g. `deepseek/deepseek-reasoner`):

- You can see both the reasoning (dim green) and final answer (bright green) text in the streamed output.
- When directly calling the LLM (without using an Agent), the `LLMResponse` object will now contain a `reasoning` field,
  in addition to the earlier `message` field.
- when using a `ChatAgent.llm_response`, extract the reasoning text from the `ChatDocument` object's `reasoning` field
  (in addition to extracting final answer as usual from the `content` field)

Below is a simple example, also in this [script](https://github.com/langroid/langroid/blob/main/examples/reasoning/agent-reasoning.py):

Some notes: 

- To get reasoning trace from Deepseek-R1 via OpenRouter, you must include
the `extra_body` parameter with `include_reasoning` as shown below.
- When using the OpenAI `o3-mini` model, you can set the `resoning_effort` parameter
  to "high", "medium" or "low" to control the reasoning effort.
- As of Feb 9, 2025, OpenAI reasoning models (o1, o1-mini, o3-mini) 
  do *not* expose reasoning trace in the API response.
  
```python
import langroid as lr
import langroid.language_models as lm

llm_config = lm.OpenAIGPTConfig(
  chat_model="deepseek/deepseek-reasoner",
  # inapplicable params are automatically removed by Langroid
  params=lm.OpenAICallParams(
    reasoning_effort="low",  # only supported by o3-mini
    # below lets you get reasoning when using openrouter/deepseek/deepseek-r1
    extra_body=dict(include_reasoning=True),
  ),
)

# (1) Direct LLM interaction
llm = lm.OpenAIGPT(llm_config)

response = llm.chat("Is 9.9 bigger than 9.11?")

# extract reasoning
print(response.reasoning)
# extract answer
print(response.message)

# (2) Using an agent
agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        llm=llm_config,
        system_message="Solve the math problem given by the user",
    )
)

response = agent.llm_response(
    """
    10 years ago, Jack's dad was 5 times as old as Jack.
    Today, Jack's dad is 40 years older than Jack.
    How old is Jack today?
    """
)

# extract reasoning
print(response.reasoning)
# extract answer
print(response.content)
```

## Displaying Reasoning in UI Callbacks

When using Langroid with UI frameworks like Chainlit, the reasoning content from LLM
responses is automatically passed to the callback methods. This allows you to display
the chain-of-thought reasoning in your UI.

The following callback methods receive a `reasoning` parameter:

- `show_llm_response(content, tools_content, is_tool, cached, language, reasoning)` -
  For non-streaming LLM responses
- `finish_llm_stream(content, tools_content, is_tool, reasoning)` -
  For streaming LLM responses

### Chainlit Integration

When using `ChainlitAgentCallbacks` or `ChainlitTaskCallbacks`, reasoning content is
automatically displayed as a nested message under the main LLM response. The reasoning
appears with a "💭 Reasoning" label in the author field.

### Custom Callback Implementation

If you're implementing custom callbacks, you can access the reasoning parameter to
display it however you prefer:

```python
from langroid.agent.base import Agent

def my_show_llm_response(
    content: str,
    tools_content: str = "",
    is_tool: bool = False,
    cached: bool = False,
    language: str | None = None,
    reasoning: str = "",
) -> None:
    # Display the main response
    print(f"Response: {content}")

    # Display reasoning if available
    if reasoning:
        print(f"Reasoning: {reasoning}")

# Attach to an agent
agent = Agent(config)
agent.callbacks.show_llm_response = my_show_llm_response
```
