"""
Single-agent question-answering system that has access to Google Search when needed,
and in case a Google Search is used, ingests contents into a vector-db,
and uses Retrieval Augmentation to answer the question.

Run like this:

    python3 examples/docqa/chat-search.py

Optional args:
    -nc : turn off caching (i.e. don't retrieve cached LLM responses)
    -d: debug mode, to show all intermediate results
    -f: use OpenAI functions api instead of tools
    -m <model_name>:  (e.g. -m litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M)
    (defaults to GPT4-Turbo if blank)

(See here for guide to using non-OpenAI LLMs:
https://langroid.github.io/langroid/tutorials/non-openai-llms/)

NOTE: running this example requires setting the GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).
"""

import re
import typer
from rich import print
from rich.prompt import Prompt

from pydantic import BaseSettings
import langroid as lr
import langroid.language_models as lm
from langroid.agent.tool_message import ToolMessage
from langroid.agent.chat_agent import ChatAgent, ChatDocument
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.parsing.web_search import google_search
from langroid.agent.task import Task
from langroid.utils.constants import NO_ANSWER
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


class RelevantExtractsTool(ToolMessage):
    request = "relevant_extracts"
    purpose = "Get docs/extracts relevant to the <query>"
    query: str


class RelevantSearchExtractsTool(ToolMessage):
    request = "relevant_search_extracts"
    purpose = "Get docs/extracts relevant to the <query> from a web search"
    query: str
    num_results: int = 3


class GoogleSearchDocChatAgent(DocChatAgent):
    tried_vecdb: bool = False

    def llm_response(
        self,
        query: None | str | ChatDocument = None,
    ) -> ChatDocument | None:
        return ChatAgent.llm_response(self, query)

    def relevant_extracts(self, msg: RelevantExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from vecdb"""
        self.tried_vecdb = True
        query = msg.query
        _, extracts = self.get_relevant_extracts(query)
        if len(extracts) == 0:
            return NO_ANSWER
        return "\n".join(str(e) for e in extracts)

    def relevant_search_extracts(self, msg: RelevantSearchExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from a web search"""
        if not self.tried_vecdb:
            return "Please try the `relevant_extracts` tool, before using this tool"
        self.tried_vecdb = False
        query = msg.query
        num_results = msg.num_results
        results = google_search(query, num_results)
        links = [r.link for r in results]
        self.config.doc_paths = links
        self.ingest()
        _, extracts = self.get_relevant_extracts(query)
        return "\n".join(str(e) for e in extracts)


class CLIOptions(BaseSettings):
    fn_api: bool = False
    model: str = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the Google Search chatbot!
        I will try to answer your questions, relying on (full content of links from) 
        Google Search when needed.
        
        Enter x or q to quit, or ? for evidence
        """
    )

    system_msg = Prompt.ask(
        """
    [blue] Tell me who I am; complete this sentence: You are...
    [or hit enter for default]
    [blue] Human
    """,
        default="a helpful assistant.",
    )
    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)

    llm_config = lm.OpenAIGPTConfig(
        chat_model=opts.model or lm.OpenAIChatModel.GPT4_TURBO,
        # or, other possibilities for example:
        # "litellm/bedrock/anthropic.claude-instant-v1"
        # "litellm/ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=2048,  # adjust based on model
    )

    config = DocChatAgentConfig(
        use_functions_api=opts.fn_api,
        use_tools=not opts.fn_api,
        llm=llm_config,
        relevance_extractor_config=lr.agent.special.RelevanceExtractorAgentConfig(
            llm=llm_config
        ),
        system_message=f"""
        {system_msg} You will try your best to answer my questions,
        in this order of preference:
        1. If you can answer from your own knowledge, simply return the answer
        2. Otherwise, ask me for some relevant text, and I will send you. Use the 
            `relevant_extracts` tool/function-call for this purpose. Once you receive 
            the text, you can use it to answer my question. 
            If I say {NO_ANSWER}, it means I found no relevant docs, and you can try 
            the next step, using a web search.
        3. If you are still unable to answer, you can use the `relevant_search_extracts`
           tool/function-call to get some text from a web search. Once you receive the
           text, you can use it to answer my question.
        4. If you still can't answer, simply say {NO_ANSWER} 
        
        Remember to always FIRST try `relevant_extracts` to see if there are already 
        any relevant docs, before trying web-search with `relevant_search_extracts`.
        
        Be very concise in your responses, use no more than 1-2 sentences.
        When you answer based on provided documents, be sure to show me 
        the SOURCE(s) and EXTRACT(s), for example:
        
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        """,
    )

    agent = GoogleSearchDocChatAgent(config)
    agent.enable_message(RelevantExtractsTool)
    agent.enable_message(RelevantSearchExtractsTool)
    collection_name = Prompt.ask(
        "Name a collection to use",
        default="docqa-chat-search",
    )
    replace = (
        Prompt.ask(
            "Would you like to replace this collection?",
            choices=["y", "n"],
            default="n",
        )
        == "y"
    )

    print(f"[red]Using {collection_name}")

    agent.vecdb.set_collection(collection_name, replace=replace)

    task = Task(agent)
    task.run("Can you help me answer some questions, possibly using web search?")


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    fn_api: bool = typer.Option(False, "--fn_api", "-f", help="use functions api"),
) -> None:
    cli_opts = CLIOptions(
        fn_api=fn_api,
        model=model,
    )

    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    chat(cli_opts)


if __name__ == "__main__":
    app()
