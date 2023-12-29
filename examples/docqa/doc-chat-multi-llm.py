"""
2-agent doc-chat:
WriterAgent (powered by GPT4) is in charge of answering user's question,
which can be complex.
Breaks it down into smaller questions (if needed) to send to DocAgent
(powered by a possibly weaker but cheaper LLM),
who has access to the docs via a vector-db.

A few ways to run this:

(a) GPT4 for both agents, WriterAgent and DocAgent:
python3 examples/docqa/doc-chat-multi-llm.py

(b) GPT4 for WriterAgent, local LLM for DocAgent, assuming you have done
`ollama run mistral`:

python3 examples/docqa/doc-chat-multi-llm.py -m litellm/ollama/mistral

(c) GPT4 for WriterAgent, local LLM for DocAgent, assuming you have done
`ollama pull mistral` and spun up the liteLLM OpenAI proxy server listening on
localhost:8000:

python3 examples/docqa/doc-chat-multi-llm.py -m local/localhost:8000

See here for a guide on how to use Langroid with non-OpenAI LLMs (local/remote):
https://langroid.github.io/langroid/tutorials/non-openai-llms/

"""
import typer
from rich import print
import os

import langroid as lr
import langroid.language_models.base
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.utils.constants import NO_ANSWER

app = typer.Typer()

setup_colored_logging()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chat(config: DocChatAgentConfig) -> None:
    doc_agent = DocChatAgent(config)
    print("[blue]Welcome to the document chatbot!")
    doc_agent.user_docs_ingest_dialog()
    print("[cyan]Enter x or q to quit, or ? for evidence")

    doc_task = Task(
        doc_agent,
        name="DocAgent",
        done_if_no_response=[lr.Entity.LLM],
        done_if_response=[lr.Entity.LLM],
    )

    writer_agent = ChatAgent(
        ChatAgentConfig(
            name="WriterAgent",
            llm=OpenAIGPTConfig(),
            vecdb=None,
        )
    )
    writer_agent.enable_message(RecipientTool)
    writer_task = Task(
        writer_agent,
        name="WriterAgent",
        system_message=f"""
        You are tenacious, creative and resourceful when given a question to 
        find an answer for. You will receive questions from a user, which you will 
        try to answer ONLY based on content from certain documents (not from your 
        general knowledge). However you do NOT have access to the documents. 
        You will be assisted by DocAgent, who DOES have access to the documents.
        
        Here are the rules:
        (a) when the question is complex or has multiple parts, break it into small 
         parts and/or steps and send them to DocAgent
        (b) if DocAgent says {NO_ANSWER} or gives no answer, try asking in other ways.
        (c) Once you collect all parts of the answer, you can say DONE and give me 
            the final answer. 
        (d) DocAgent has no memory of previous dialog, so you must ensure your 
            questions are stand-alone questions that don't refer to entities mentioned 
            earlier in the dialog.
        (e) if DocAgent is unable to answer after your best efforts, you can say
            {NO_ANSWER} and move on to the next question.
        (f) answers should be based ONLY on the documents, NOT on your prior knowledge.
        (g) be direct and concise, do not waste words being polite.
        (h) if you need more info from the user, before asking DocAgent, you should 
        address questions to the "User" (not to DocAgent) to get further 
        clarifications or information. 
        (i) Always ask questions ONE BY ONE (to either User or DocAgent), NEVER 
            send Multiple questions in one message.
        (j) Use bullet-point format when presenting multiple pieces of info.
        (k) When DocAgent responds without citing a SOURCE and EXTRACT(S), you should
            send your question again to DocChat, reminding it to cite the source and
            extract(s).
        
        
        Start by asking the user what they want to know.
        """,
    )
    writer_task.add_sub_task(doc_task)
    writer_task.run()

    # show cost summary
    print("LLM usage, cost summary:")
    print(str(langroid.language_models.base.LanguageModel.usage_cost_summary()))


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm")
    my_llm_config = MyLLMConfig(
        chat_model="litellm/ollama/llama2",
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "litellm/ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=8192,  # adjust based on model
        timeout=45,
    )

    if model == "":
        llm_config = OpenAIGPTConfig(timeout=45)  # default GPT-4
    else:
        llm_config = my_llm_config
        llm_config.chat_model = model

    config = DocChatAgentConfig(
        llm=llm_config,
        n_query_rephrases=0,
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        hypothetical_answer=False,
        assistant_mode=True,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=500,  # aim for this many tokens per chunk
            overlap=200,  # overlap between chunks
            max_chunks=10_000,
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            n_similar_docs=5,
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
            cache_type=cache_type,
        )
    )
    chat(config)


if __name__ == "__main__":
    app()
