"""
Document question-answering using RAG on a single file, using ChainlitAgentCallbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-doc-qa.py

Note, to run this with a local LLM, you can click the settings symbol
on the left of the chat window and enter the model name, e.g.:

ollama/mistral:7b-instruct-v0.2-q8_0

or

local/localhost:8000/v1"

depending on how you have set up your local LLM.

For more on how to set up a local LLM to work with Langroid, see:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

"""

from textwrap import dedent

import chainlit as cl

import langroid as lr
import langroid.parsing.parser as lp
from langroid.agent.callbacks.chainlit import (
    SYSTEM,
    add_instructions,
    get_text_files,
    make_llm_settings_widgets,
    setup_llm,
    update_llm,
)
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.utils.constants import NO_ANSWER


async def initialize_agent() -> None:
    await setup_llm()
    llm_config = cl.user_session.get("llm_config")
    config = DocChatAgentConfig(
        name="DocAgent",
        n_query_rephrases=0,
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=0,
        n_similar_chunks=3,
        n_relevant_chunks=3,
        llm=llm_config,
        parsing=lp.ParsingConfig(  # modify as needed
            splitter=lp.Splitter.TOKENS,
            chunk_size=300,  # aim for this many tokens per chunk
            overlap=30,  # overlap between chunks
            max_chunks=10_000,
            n_neighbor_ids=5,  # store ids of window of k chunks around each chunk.
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=lp.PdfParsingConfig(
                # alternatives: "unstructured", "docling", "fitz"
                library="pymupdf4llm",
            ),
        ),
    )
    agent = DocChatAgent(config)
    cl.user_session.set("agent", agent)


@cl.on_settings_update
async def on_update(settings):
    await update_llm(settings)
    await initialize_agent()


@cl.on_chat_start
async def on_chat_start():
    await add_instructions(
        title="Basic Doc-Question-Answering using RAG (Retrieval Augmented Generation).",
        content=dedent(
            """
        **Upload** a document (click the attachment button in the chat dialog) and ask questions.
        **Change LLM settings** by clicking the settings symbol on the left of the chat window.
        
        You can keep uploading more documents, and questions will be answered based on all documents.
        """
        ),
    )

    await make_llm_settings_widgets()

    cl.user_session.set("callbacks_inserted", False)
    await initialize_agent()


@cl.on_message
async def on_message(message: cl.Message):
    agent: DocChatAgent = cl.user_session.get("agent")
    file2path = await get_text_files(message)
    agent.callbacks.show_start_response(entity="llm")
    if len(file2path) > 0:
        n_files = len(file2path)
        waiting = cl.Message(
            author=SYSTEM, content=f"Received {n_files} files. Ingesting..."
        )
        await waiting.send()
        agent.ingest_doc_paths(list(file2path.values()))
        file_or_files = "file" if n_files == 1 else "files"
        file_list = "\n".join([f"- `{file}`" for file in file2path.keys()])
        waiting.content = dedent(
            f"""
            Ingested `{n_files}` {file_or_files}:
            {file_list}
            """
        )
        await waiting.update()

    if not cl.user_session.get("callbacks_inserted", False):
        # first time user entered a msg, so inject callbacks and display first msg
        lr.ChainlitAgentCallbacks(agent)

    # Note DocChatAgent has no llm_response_async,
    # so we use llm_response with make_async
    response: lr.ChatDocument | None = await cl.make_async(agent.llm_response)(
        message.content
    )
    if response.content.strip() == NO_ANSWER:
        # in this case there were no relevant extracts
        # and we never called the LLM, so response was not shown in UI,
        # hence we need to send it here
        # TODO: It is possible the LLM might have already responded with NO_ANSWER,
        # so we may be duplicating the response here.
        agent.callbacks.show_llm_response(content=NO_ANSWER)
