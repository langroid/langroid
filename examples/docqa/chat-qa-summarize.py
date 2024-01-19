"""
Two-agent system to do Question-Answer based summarization of documents.
E.g. one could use this to summarize a very large document, assuming there is a
reasonable abstract/intro at the start that "covers" the import aspects.

WriterAgent (has no access to docs) is tasked with writing 5 bullet points based on
some docs. Initially it generates a summary of the docs from the beginning of the doc,
then it formulates questions to ask until it gets 5 key pieces of information.

DocAgent (has access to docs) answers these questions using RAG.

Run like this:

python examples/docqa/chat-qa-summarize.py

You can let it run and it will finish with 5 key bullet points about the document(s).

There are optional args, especially note you can pass in a different LLM model, e.g.

python examples/docqa/chat-qa-summarize.py -m litellm/ollama/mistral

[To avail of this, remember to install the litellm extra, e.g.
`pip install "langroid[litellm]"` or `poetry add langroid[litellm]`]

See this example for more details on how to specify a different LLM model:
https://github.com/langroid/langroid-examples/blob/main/examples/docqa/rag-local-simple.py

"""
import typer
from rich import print
import os

from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
import langroid.language_models as lm
from langroid.mytypes import Entity
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.parsing.urls import get_list_from_user
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="specify alternative LLM, e.g. litellm/ollama/mistral",
    ),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    config = DocChatAgentConfig(
        llm=lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        ),
    )
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            cache_type=cache_type,
        )
    )
    doc_agent = DocChatAgent(config)
    doc_agent.vecdb.set_collection("docqa-chat-multi", replace=True)
    print("[blue]Welcome to the document chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print(
        """
        [blue]Enter some URLs or file/dir paths below (or leave empty for default URLs)
        """.strip()
    )
    inputs = get_list_from_user()
    if len(inputs) == 0:
        inputs = config.default_paths
    doc_agent.config.doc_paths = inputs
    doc_agent.ingest()
    topics_doc = doc_agent.summarize_docs(
        instruction="""
        Give me a list of up to 3 main topics from the following text,
        in the form of short sentences.
        """,
    )
    topics = topics_doc.content
    doc_task = Task(
        doc_agent,
        name="DocAgent",
        done_if_no_response=[Entity.LLM],  # done if null response from LLM
        done_if_response=[Entity.LLM],  # done if non-null response from LLM
        system_message="""You will receive various questions about some documents, and
        your job is to answer them concisely in at most 2 sentences, citing sources.
        """,
    )

    writer_agent = ChatAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(),
            vecdb=None,
        )
    )
    writer_task = Task(
        writer_agent,
        # SET interactive to True to slow it down, but keep hitting enter to progress
        interactive=False,
        name="WriterAgent",
        system_message=f"""
        You have to collect some information from some documents, on these topics:
        {topics}
        However you do not have access to those documents. 
        You can ask me questions about them, ONE AT A TIME, I will answer each 
        question. 
        Once you have collected 5 key pieces of information, say "DONE" and summarize 
        them in bullet points.  
        """,
    )
    writer_task.add_sub_task(doc_task)
    writer_task.run()


if __name__ == "__main__":
    app()
