"""
Single agent to use to chat with an LLM using  Retrieval-Augmented Generation (RAG).
Similar to chat.py but allows specifying a local LLM.
You must either have:
 - (A) a local LLM running at an OpenAI-compatible API endpoint, e.g. localhost:8000, or
 - (B) use a local-LLM library supported by litellm, e.g. ollama, and do `ollama pull mistral`,
     (for this scenario you have to have the `litellm` extra installed,
     via `pip install langroid[ollama]` or equivalent)

For scenario (A) you can specify the local model on the cmd line for example as:
`-m "local/localhost:8000"`
And for scenario (B) you can specify the local model on the cmd line for example as:
`-m "litellm/ollama/mistral"`

See docs here: https://langroid.github.io/langroid/tutorials/non-openai-llms/

Run like this for example in scenario (A):

python3 examples/docqa/chat-local.py -m "local/localhost:8000"

and for scenario (B):

python3 examples/docqa/chat-local.py -m "litellm/ollama/llama2"

CAVEAT:
(1) The app works best with GPT4/Turbo, but results may be mixed with local LLMs.
You may have to tweak the system_message, use_message, and summarize_prompt
as indicated in comments below, to get good results.
(2) The default vector-db in DocChatAgent is LanceDB, but you can switch to the other
    supported vector-dbs, e.g. qdrant or chroma.

"""
import re
import typer
from rich import print
from rich.prompt import Prompt
import os

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
) -> None:
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "litellm/ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=2048,  # adjust based on model
    )

    relevance_extractor_config = lr.agent.special.RelevanceExtractorAgentConfig(
        llm=llm_config,  # or this could be a different llm_config
        # system_message="...override default RelevanceExtractorAgent system msg here",
    )

    config = DocChatAgentConfig(
        n_query_rephrases=0,
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=0,
        llm=llm_config,
        relevance_extractor_config=relevance_extractor_config,  # or None to turn off
        # system_message="...override default DocChatAgent system msg here",
        # user_message="...override default DocChatAgent user msg here",
        # summarize_prompt="...override default DocChatAgent summarize prompt here",
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=300,  # aim for this many tokens per chunk
            overlap=30,  # overlap between chunks
            max_chunks=10_000,
            n_neighbor_ids=5,  # store ids of window of k chunks around each chunk.
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            n_similar_docs=3,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "haystack", "unstructured", "pdfplumber", "fitz"
                library="pdfplumber",
            ),
        ),
    )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    agent = DocChatAgent(config)
    print("[blue]Welcome to the document chatbot!")
    agent.user_docs_ingest_dialog()
    print("[cyan]Enter x or q to quit, or ? for evidence")

    system_msg = Prompt.ask(
        """
    [blue] Tell me who I am; complete this sentence: You are...
    [or hit enter for default] 
    [blue] Human
    """,
        default="a helpful assistant.",
    )
    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)
    task = Task(
        agent,
        system_message="You are " + system_msg,
    )
    task.run()


if __name__ == "__main__":
    app()
