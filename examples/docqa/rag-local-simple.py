"""
RAG example using a local LLM, with ollama

(1) Mac: Install latest ollama, then do this:
ollama pull mistral:7b-instruct-v0.2-q4_K_M

(2) Ensure you've installed the `litellm` extra with Langroid, e.g.
pip install langroid[litellm], or if you use the `pyproject.toml` in this repo
you can simply use `poetry install`

(3) Run like this:

python3 examples/docqa/rag-local-simple.py

To change the local model, use the optional arg -m <local_model>.

Other possibilities for local_model are:
- If instead of ollama (perhaps using oobo text-generation-webui)
  you've spun up your local LLM to listen at an OpenAI-Compatible Endpoint
  like `localhost:8000`, then you can use `-m local/localhost:8000`
- If the endpoint is listening at https://localhost:8000/v1, you must include the `v1`
- If the endpoint is http://127.0.0.1:8000, use `-m local/127.0.0.1:8000`

And so on. The above are few-shot examples for you. You get the idea!
"""

import os
import fire
import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def app(
    m="litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M",
):
    # Create the llm config object.
    llm_config = lm.OpenAIGPTConfig(
        # if you comment out `chat_model`, it will default to OpenAI GPT4-turbo
        # chat_model="litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M",
        chat_model=m or lm.OpenAIChatModel.GPT4_TURBO,
        chat_context_length=4096,  # set this based on model
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
        # default vector-db is LanceDB,
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
