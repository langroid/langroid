"""
Simple example showing how you can separately
extract the reasoning (thinking) and final response from a langroid ChatAgent

Run like this (omit the model argument to default to the deepseek-reasoner model):

    python examples/reasoning/agent-reasoning.py \
    --model  gemini/gemini-2.0-flash-thinking-exp

or
    uv run examples/reasoning/agent-reasoning.py

Other reasoning models to try:
openrouter/deepseek/deepseek-r1
o1
o1-mini
o3-mini
ollama/deepseek-r1:8b

"""

import langroid as lr
import langroid.language_models as lm
from langroid.utils.configuration import settings
from fire import Fire


def main(
    model: str = "",
    nc: bool = False,  # turn off caching? (i.e. get fresh streaming response)
):
    settings.cache = not nc
    model = model or "deepseek/deepseek-reasoner"
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model,
        # inapplicable params are automatically removed by Langroid
        params=lm.OpenAICallParams(
            reasoning_effort="low",  # only supported by o3-mini
            # below lets you get reasoning when using openrouter/deepseek/deepseek-r1
            extra_body=dict(include_reasoning=True),
        ),
    )

    # (1) Direct LLM interaction
    llm = lm.OpenAIGPT(llm_config)

    response = llm.chat("Is 9.3 bigger than 9.11?", max_tokens=1000)

    if response.cached:
        # if we got it from cache, we haven't shown anything, so print here

        # extract reasoning
        if response.reasoning:
            print(response.reasoning)
        else:
            print(f"NO REASONING AVAILABLE for {model}!")
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

    if response.metadata.cached:
        # if we got it from cache, we haven't shown anything, so print here

        # extract reasoning
        if response.reasoning:
            print(response.reasoning)
        else:
            print(f"NO REASONING AVAILABLE for {model}!")

        # extract answer
        print(response.content)


if __name__ == "__main__":
    Fire(main)
