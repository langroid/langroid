"""
Test multi-round interaction with a local LLM, playing a simple "doubling game".

In each round:

- User gives a number
- LLM responds with the double of that number

Run like this --

python3 examples/basic/chat-local-numerical.py -m <model_name_with_formatter_after//>

Recommended local model setup:
- spin up an LLM with oobabooga at an endpoint like http://127.0.0.1:5000/v1
- run this script with -m local/127.0.0.1:5000/v1
- To ensure accurate chat formatting (and not use the defaults from ooba),
  append the appropriate HuggingFace model name to the
  -m arg, separated by //, e.g. -m local/127.0.0.1:5000/v1//mistral-instruct-v0.2
  (no need to include the full model name, as long as you include enough to
   uniquely identify the model's chat formatting template)

"""
import os
import fire

import langroid as lr
from langroid.utils.configuration import settings
import langroid.language_models as lm

# for best results:
DEFAULT_LLM = lm.OpenAIChatModel.GPT4_TURBO

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# (1) Define the desired fn-call as a ToolMessage via Pydantic.


def app(
    m: str = DEFAULT_LLM,  # model name
    d: bool = False,  # debug
    nc: bool = False,  # no cache
):
    settings.debug = d
    settings.cache = not nc
    # create LLM config
    llm_cfg = lm.OpenAIGPTConfig(
        chat_model=m or DEFAULT_LLM,
        chat_context_length=4096,  # set this based on model
        max_output_tokens=100,
        temperature=0.2,
        stream=True,
        timeout=45,
    )

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_cfg,
            system_message="""
            You are a number-doubling expert. When user gives you a NUMBER,
            simply respond with its DOUBLE and SAY NOTHING ELSE.
            DO NOT EXPLAIN YOUR ANSWER OR YOUR THOUGHT PROCESS.
            """,
        )
    )

    task = lr.Task(agent)
    task.run("15")  # initial number


if __name__ == "__main__":
    fire.Fire(app)
