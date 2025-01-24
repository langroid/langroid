"""
Simple example showing how you can separately
extract the reasoning (thinking) and final response from a langroid ChatAgent

Run like this:

    python examples/reasoning/agent-reasoning.py

or
    uv run examples/reasoning/agent-reasoning.py
"""

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


# (2) Agent interaction
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
