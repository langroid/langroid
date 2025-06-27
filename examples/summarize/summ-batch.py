"""
Batch version of summ.py.

Summarize a collection of docs, loaded into context, using a local LLM, with ollama.
First see instructions to install langroid
in the README of the langroid-examples repo:
https://github.com/langroid/langroid-examples

Run like this from the root of the project repo:

python3 examples/summarize/summ-batch.py -m <model_name>

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

PATH = "examples/summarize/data/hf-cnn-daily-news/news10.csv"


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
    # get column "article" as list of strings, from first few rows
    full_docs = [str(row) for row in df["article"][:10]]
    # get column "highlights" as list of strings, from first few rows
    highlights = [str(row) for row in df["highlights"][:10]]

    print(f"Found {len(full_docs)} documents to summarize.")

    config = lr.ChatAgentConfig(
        llm=llm_config,
        system_message="""
        You are an expert in finding the main points in a document,
        and generating concise summaries of them.
        When user gives you a document, summarize it in at most 3 sentences.
        """,
    )

    agent = lr.ChatAgent(config)
    summaries = lr.llm_response_batch(
        agent,
        full_docs,
        output_map=lambda x: x.content,
    )

    for i, summary in enumerate(summaries):
        print(
            f"""
        Generated Summary {i}:
        {summary}
        """
        )

        print(
            f"""
        Gold Summary {i}:
        {highlights[i]}
        """
        )


if __name__ == "__main__":
    fire.Fire(app)
