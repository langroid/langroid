"""
RAG example using a local LLM, with ollama

# (1) Mac: Install latest ollama, then do this:
# ollama pull mistral:7b-instruct-v0.2-q4_K_M

# (2) Ensure you've installed the `litellm` extra with Langroid, e.g.
# pip install langroid[litellm], or if you use the `pyproject.toml` in this repo
# you can simply use `poetry install`

# (3) Run like this:

python3 examples/docqa/rag-local-simple.py

"""
import os
import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create the llm config object.
# Note: if instead of ollama you've spun up your local LLM to listen at
# an OpenAI-Compatible Endpoint like `localhost:8000`, then you can set
# chat_model="local/localhost:8000"; carefully note there's no http in this,
# and if the endpoint is localhost:8000/v1, then you must set
# chat_model="local/localhost:8000/v1"
# Similarly if your endpoint is `http://128.0.4.5:8000/v1`, then you must set
# chat_model="local/128.0.4.5:8000/v1"
llm_config = lm.OpenAIGPTConfig(
    chat_model="litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M",
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

hf_embed_config = lr.embedding_models.SentenceTransformerEmbeddingsConfig(
    model_type="sentence-transformer",
    model_name="BAAI/bge-large-en-v1.5",
)

config = DocChatAgentConfig(
    llm=llm_config,
    relevance_extractor_config=lr.agent.special.RelevanceExtractorAgentConfig(
        llm=llm_config
    ),
    doc_paths=[
        # can be URLS, file-paths, or Folders.
        # File-types: most web-pages, and local pdf, txt, docx
        "https://arxiv.org/pdf/2312.17238.pdf",
    ],
    system_message="""
    Answer some questions about docs. Be concise. 
    Start by asking me what I want to know
    """,
)

agent = DocChatAgent(config)
task = lr.Task(agent)
task.run()
