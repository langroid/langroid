"""
RAG example using a local LLM, with ollama

Run like this --

python3 examples/docqa/rag-local-simple.py -m <model_name>

For example, you can get good results using:
```
ollama run mistral:7b-instruct-v0.2-q8_0

python3 examples/docqa/rag-local-simple.py -m ollama/mistral:7b-instruct-v0.2-q8_0


See here for more on how to set up a local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import os

import fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def app(m="ollama/mistral:7b-instruct-v0.2-q8_0"):
    # Create the llm config object.
    llm_config = lm.OpenAIGPTConfig(
        # if you comment out `chat_model`, it will default to OpenAI GPT4-turbo
        # chat_model="ollama/mistral:7b-instruct-v0.2-q4_K_M",
        chat_model=m or lm.OpenAIChatModel.GPT4o,
        chat_context_length=32_000,  # set this based on model
        max_output_tokens=100,
        temperature=0.2,
        stream=True,
        timeout=45,
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
    # task = lr.Task(agent)
    # verify you can interact with this in a chat loop on cmd line:
    # task.run("Concisely answer some questions")

    config = DocChatAgentConfig(
        # default vecdb is qdrantdb
        # using SentenceTransformers/BAAI/bge-large-en-v1.5 embedding model
        llm=llm_config,
        doc_paths=[
            # can be URLS, file-paths, or Folders.
            # File-types: most web-pages, and local pdf, txt, docx
            "https://arxiv.org/pdf/2312.17238.pdf",
        ],
        system_message="""
        Concisely answer my questions about docs. Start by asking me what I want to know.
        """,
    )

    agent = DocChatAgent(config)
    task = lr.Task(agent)
    task.run()


if __name__ == "__main__":
    fire.Fire(app)
