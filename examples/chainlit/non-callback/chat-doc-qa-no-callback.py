"""
Basic single-agent chat example, without streaming.

DEPCRECATED: Script kept only for reference. Best way is to use ChainlitAgentCallbacks,
as in chat-doc-qa.py

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-doc-qa-no-callback.py

Note, to run this with a local LLM, you can click the settings symbol
on the left of the chat window and enter the model name, e.g.:

ollama/mistral:7b-instruct-v0.2-q8_0

or

local/localhost:8000/v1"

depending on how you have set up your local LLM.

For more on how to set up a local LLM to work with Langroid, see:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

"""

import chainlit as cl
import langroid.parsing.parser as lp
import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig


async def setup_agent() -> None:
    model = cl.user_session.get("settings", {}).get("ModelName")
    print(f"Using model: {model}")
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=16_000,  # adjust based on model
        timeout=90,
    )

    config = DocChatAgentConfig(
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
    file = cl.user_session.get("file")
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()
    agent.ingest_doc_paths([file.path])
    msg.content = f"Processing `{file.name}` done. Ask questions!"
    await msg.update()


@cl.on_settings_update
async def update_agent(settings):
    cl.user_session.set("settings", settings)
    await setup_agent()


@cl.on_chat_start
async def on_chat_start():
    await cl.ChatSettings(
        [
            cl.input_widget.TextInput(
                id="ModelName",
                label="Model Name (Default GPT4-Turbo)",
                default="",
            )
        ]
    ).send()

    # get file
    files = None
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    print(f"got file: {file.name}")
    cl.user_session.set("file", file)
    await setup_agent()


@cl.on_message
async def on_message(message: cl.Message):
    agent: DocChatAgent = cl.user_session.get("agent")
    msg = cl.Message(content="")

    # need to do this since DocChatAgent does not have an async version of llm_response
    response = await cl.make_async(agent.llm_response)(message.content)
    msg.content = response.content
    await msg.send()
