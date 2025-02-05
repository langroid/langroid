# Stream and capture reasoning content in addition to final answer, from Reasoning LLMs

As of v0.35.0, when using Reasoning LLM APIs (e.g. `deepseek/deepseek-reasoner`
or OpenAI `o1` etc):

- You can see both the reasoning (dim green) and final answer (bright green) text in the streamed output.
- When directly calling the LLM (without using an Agent), the `LLMResponse` object will now contain a `reasoning` field,
  in addition to the earlier `message` field.
- when using a `ChatAgent.llm_response`, extract the reasoning text from the `ChatDocument` object's `reasoning` field
  (in addition to extracting final answer as usual from the `content` field)

Here's a simple example, also in this [script](https://github.com/langroid/langroid/blob/main/examples/reasoning/agent-reasoning.py):

```python
import langroid as lr
import langroid.language_models as lm

llm_config = lm.OpenAIGPTConfig(chat_model="deepseek/deepseek-reasoner")

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
