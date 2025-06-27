"""
2-Agent system where:
- Assistant takes user's (complex) question, breaks it down into smaller pieces
    if needed
- WebSearcher takes Assistant's question, uses the Search tool to search the web
    (default DuckDuckGo, or Google or Metaphor as specified by user), and returns a
    coherent answer to the Assistant.

Once the Assistant thinks it has enough info to answer the user's question, it
says DONE and presents the answer to the user.

See also: chat-search for a basic single-agent search

python3 examples/basic/chat-search-assistant.py

There are optional args, especially note these:

-p or --provider: google or ddg or metaphor (default: google)
-m <model_name>: to run with a different LLM model (default: gpt4-turbo)

You can specify a local in a few different ways, e.g. `-m local/localhost:8000/v1`
or `-m ollama/mistral` etc. See here how to use Langroid with local LLMs:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


NOTE:
(a) If using Google Search, you must have GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).

(b) If using MetaphorSearchTool, you need to:
* set the METAPHOR_API_KEY environment variables in
your `.env` file, e.g. `METAPHOR_API_KEY=your_api_key_here`
* install langroid with the `metaphor` extra, e.g.
`pip install langroid[metaphor]` or `uv pip install langroid[metaphor]`
`poetry add langroid[metaphor]` or `uv add langroid[metaphor]`
(it installs the `metaphor-python` package from pypi).
For more information, please refer to the official docs:
https://metaphor.systems/

"""

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    provider: str = typer.Option(
        "ddg",
        "--provider",
        "-p",
        help="search provider name (google, metaphor, ddg)",
    ),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    print(
        """
        [blue]Welcome to the Web Search Assistant chatbot!
        I will try to answer your complex questions. 
        
        Enter x or q to quit at any point.
        """
    )
    load_dotenv()

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=8_000,
        temperature=0,
        max_output_tokens=200,
        timeout=45,
    )

    assistant_config = lr.ChatAgentConfig(
        system_message=f"""
        You are a resourceful assistant, able to think step by step to answer
        complex questions from the user. You must break down complex questions into
        simpler questions that can be answered by a web search. You must ask me 
        (the user) each question ONE BY ONE, and I will do a web search and send you
        a brief answer. Once you have enough information to answer my original
        (complex) question, you MUST say {DONE} and present the answer to me.
        """,
        llm=llm_config,
        vecdb=None,
    )
    assistant_agent = lr.ChatAgent(assistant_config)

    match provider:
        case "google":
            search_tool_class = GoogleSearchTool
        case "metaphor":
            from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool

            search_tool_class = MetaphorSearchTool
        case "ddg":
            search_tool_class = DuckduckgoSearchTool
        case _:
            raise ValueError(f"Unsupported provider {provider} specified.")

    search_tool_handler_method = search_tool_class.name()

    search_agent_config = lr.ChatAgentConfig(
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a web-searcher. For any question you get, you must use the TOOL
        `{search_tool_handler_method}`  to get up to 5 results.
        I WILL SEND YOU THE RESULTS; DO NOT MAKE UP THE RESULTS!!
        Once you receive the results, you must compose a CONCISE answer 
        based on the search results and say {DONE} and show the answer to me,
        in this format:
        {DONE} [... your CONCISE answer here ...]
        IMPORTANT:
        * YOU MUST WAIT FOR ME TO SEND YOU THE SEARCH RESULTS BEFORE saying  {DONE}.
        * YOU Can only use the TOOL `{search_tool_handler_method}` 
            ONE AT A TIME, even if you get multiple questions!
        """,
    )
    search_agent = lr.ChatAgent(search_agent_config)
    search_agent.enable_message(search_tool_class)

    assistant_task = lr.Task(
        assistant_agent,
        name="Assistant",
        llm_delegate=True,
        single_round=False,
        interactive=False,
    )
    search_task = lr.Task(
        search_agent,
        name="Searcher",
        llm_delegate=True,
        single_round=False,
        interactive=False,
    )
    assistant_task.add_sub_task(search_task)
    question = Prompt.ask("What do you want to know?")
    assistant_task.run(question)


if __name__ == "__main__":
    app()
