"""
Document question-answering using RAG on a single file, using ChainlitAgentCallbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-doc-qa.py

Note, to run this with a local LLM, you can click the settings symbol
on the left of the chat window and enter the model name, e.g.:

litellm/ollama_chat/mistral:7b-instruct-v0.2-q8_0

or

local/localhost:8000/v1"

depending on how you have set up your local LLM.

For more on how to set up a local LLM to work with Langroid, see:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

"""

import chainlit as cl
import langroid as lr
import langroid.parsing.parser as lp
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from langroid.agent.callbacks.chainlit import (
    add_instructions,
    make_llm_settings_widgets,
    setup_llm,
    update_agent,
)
from textwrap import dedent


async def setup_agent() -> None:
    await setup_llm()
    llm_config = cl.user_session.get("llm_config")
    config = DocChatAgentConfig(
        n_query_rephrases=0,
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        hypothetical_answer=False,
        # set it to > 0 to retrieve a window of k chunks on either side of a match
        n_neighbor_chunks=0,
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
            n_similar_docs=3,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=lp.PdfParsingConfig(
                # alternatives: "haystack", "unstructured", "pdfplumber", "fitz"
                library="pdfplumber",
            ),
        ),
    )
    agent = DocChatAgent(config)

    file = cl.user_session.get("file")
    msg = cl.Message(content="")
    await msg.send()
    agent.ingest_doc_paths([file.path])
    msg.content = f"Processing `{file.name}` done. Ask questions!"
    await msg.update()

    lr.ChainlitAgentCallbacks(agent)
    cl.user_session.set("agent", agent)


@cl.on_settings_update
async def on_update(settings):
    await update_agent(settings)


@cl.on_chat_start
async def on_chat_start():
    await add_instructions(
        title="Basic Doc-Question-Answering using RAG (Retrieval Augmented Generation).",
        content=dedent(
            """
        Upload a document and ask questions.
        Change LLM settings by clicking the settings symbol on the 
        left of the chat window.
        """
        ),
    )

    await make_llm_settings_widgets()

    # get file
    files = None
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin! (`txt`, `pdf`, `doc`, `docx`)",
            accept=[
                "text/plain",
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    print(f"got file: {file.name}")
    cl.user_session.set("file", file)
    await setup_agent()


@cl.on_message
async def on_message(message: cl.Message):
    agent: lr.ChatAgent = cl.user_session.get("agent")
    await cl.make_async(agent.llm_response)(message.content)
