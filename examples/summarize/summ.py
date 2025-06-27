"""
Summarize a doc, loaded into context, using a local LLM, with ollama.
First see instructions to install langroid
in the README of the langroid-examples repo:
https://github.com/langroid/langroid-examples

Run like this from the root of the project repo:

python3 examples/summarize/summ.py -m <model_name>

Omitting -m will use the default model, which is OpenAI GPT4-turbo.

A local LLM can be specified as follows:
```
python3 examples/summarize/summ.py -m ollama/mistral:7b-instruct-v0.2-q8_0
```

See here for more details on how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import os

import fire
import pandas as pd

import langroid as lr
import langroid.language_models as lm
from langroid.utils.configuration import settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PATH = "examples/summarize/data/news.csv"


def app(
    m: str = "",  # ollama/mistral:7b-instruct-v0.2-q8_0",
    d: bool = False,  # debug
):
    settings.debug = d
    # Create the llm config object.
    llm_config = lm.OpenAIGPTConfig(
        # if you comment out `chat_model`, it will default to OpenAI GPT4-turbo
        # chat_model="ollama/mistral:7b-instruct-v0.2-q4_K_M",
        chat_model=m or lm.OpenAIChatModel.GPT4o,
        chat_context_length=32_000,  # set this based on model
        max_output_tokens=500,  # increase this if you want longer summaries
        temperature=0.2,  # lower -> less variability
        stream=True,
        timeout=45,  # increase if model is timing out
    )

    # Recommended: First test if basic chat works with this llm setup as below:
    # Once this works, then you can try the DocChatAgent
    #
    # agent = lr.ChatAgent(
    #     lr.ChatAgentConfig(
    #         llm=llm
    #     )
    # )
    #
    # agent.llm_response("What is 3 + 4?")
    #

    df = pd.read_csv(PATH)
    full_doc = str(df["article"][0])
    highlights = str(df["highlights"][0])
    config = lr.ChatAgentConfig(
        llm=llm_config,
        system_message=f"""
        You are an expert in finding the main points in a document,
        and generating concise summaries of them.
        Summarize the article below in at most 3 (THREE) sentences:
        
        {full_doc}
        """,
    )

    agent = lr.ChatAgent(config)
    summary = agent.llm_response()
    print(
        f"""
    Generated Summary: 
    {summary.content}
    """
    )

    print(
        f"""
    Gold Summary:
    {highlights}
    """
    )


if __name__ == "__main__":
    fire.Fire(app)
