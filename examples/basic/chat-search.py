"""
This is a basic example of a chatbot that uses the GoogleSearchTool
or SciPhiSearchRAGTool to answer questions.
When the LLM doesn't know the answer to a question, it will use the tool to
search the web for relevant results, and then use the results to answer the
question.

Run like this:

python3 examples/basic/chat-search.py

There are optional args, especially note these:

-p or --provider: google or sciphi (default: google)
-m <model_name>: to run with a different LLM model (default: gpt4-turbo)

See the comments at the top of this script for more on how to specify local LLMs:
https://github.com/langroid/langroid/blob/main/examples/docqa/rag-local-simple.py



NOTE:
(a) If using Google Search, you must have GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).

(b) Alternatively, you can use the SciPhiSearchRAGTool, you need to have the
SCIPHI_API_KEY environment variable in your `.env` file.
See here for more info: https://www.sciphi.ai/
This tool requires installing langroid with the `sciphi` extra, e.g.
`pip install langroid[sciphi]` or `poetry add langroid[sciphi]`
(it installs the `agent-search` package from pypi).
"""

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    provider: str = typer.Option(
        "google", "--provider", "-p", help="search provider name (Google, SciPhi)"
    ),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    print(
        """
        [blue]Welcome to the Web Search chatbot!
        I will try to answer your questions, relying on (summaries of links from) 
        Search when needed.
        
        Enter x or q to quit at any point.
        """
    )
    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default="Default: 'You are a helpful assistant'",
    )

    load_dotenv()

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
    )

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=llm_config,
        vecdb=None,
    )
    agent = ChatAgent(config)

    match provider:
        case "google":
            search_tool_class = GoogleSearchTool
        case "sciphi":
            from langroid.agent.tools.sciphi_search_rag_tool import SciPhiSearchRAGTool

            search_tool_class = SciPhiSearchRAGTool
        case "metaphor":
            from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool

            search_tool_class = MetaphorSearchTool
        case _:
            raise ValueError(f"Unsupported provider {provider} specified.")

    agent.enable_message(search_tool_class)
    search_tool_handler_method = search_tool_class.default_value("request")

    task = Task(
        agent,
        system_message=f"""
        You are a helpful assistant. You will try your best to answer my questions.
        If you cannot answer from your own knowledge, you can use up to 5 
        results from the {search_tool_handler_method} tool/function-call to help 
        you with answering the question.
        Be very concise in your responses, use no more than 1-2 sentences.
        When you answer based on a web search, First show me your answer, 
        and then show me the SOURCE(s) and EXTRACT(s) to justify your answer,
        in this format:
        
        <your answer here>
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        SOURCE: ...
        EXTRACT: ...
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        """,
    )
    # local models do not like the first message to be empty
    user_message = "Hello." if (model != "") else None
    task.run(user_message)


if __name__ == "__main__":
    app()
