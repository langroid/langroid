"""
This is a basic example of a chatbot that uses SeltzSearchTool to
answer questions with context-engineered web content and sources
for real-time AI reasoning.

Seltz provides fast, up-to-date web data optimized for LLM consumption.

Run like this:

    python3 examples/basic/chat-search-seltz.py

There are optional args:
-m <model_name>: to run with a different LLM model (default: gpt4o)
-d: debug mode
-ns: no streaming
-nc: don't use cache

NOTE: You need to:
* set the SELTZ_API_KEY environment variable in
your `.env` file, e.g. `SELTZ_API_KEY=your_api_key_here`

* install langroid with the `seltz` extra, e.g.
`pip install langroid[seltz]` or `uv pip install langroid[seltz]`
or `poetry add langroid[seltz]` or `uv add langroid[seltz]`

For more information, please refer to https://seltz.ai/?utm_source=langroid&utm_medium=integration
"""

import typer
from dotenv import load_dotenv
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.seltz_search_tool import SeltzSearchTool
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    print(
        """
        [blue]Welcome to the Seltz Web Search chatbot!
        I will try to answer your questions using context-engineered web content
        and sources for real-time AI reasoning, powered by Seltz.

        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=32_000,
        temperature=0.15,
        max_output_tokens=1000,
        timeout=45,
    )

    search_tool_handler_method = SeltzSearchTool.name()
    config = lr.ChatAgentConfig(
        name="Seeker",
        handle_llm_no_tool="user",
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a helpful assistant. You will try your best to answer my questions.
        Here is how you should answer my questions:
        - IF my question is about a topic you ARE CERTAIN about, answer it directly
        - OTHERWISE, use the `{search_tool_handler_method}` tool/function-call to
          get up to 5 results from a web-search, to help you answer the question.
          I will show you the results from the web-search, and you can use those
          to answer the question.
        - If I EXPLICITLY ask you to search the web/internet, then use the
            `{search_tool_handler_method}` tool/function-call to get up to 5 results
            from a web-search, to help you answer the question.

        In case you use the TOOL `{search_tool_handler_method}`, you MUST WAIT
        for results from this tool; do not make up results!

        Be very CONCISE in your answers, use no more than 1-2 sentences.
        When you answer based on a web search, First show me your answer,
        and then show me the SOURCE(s) and EXTRACT(s) to justify your answer,
        in this format:

        <your answer here>
        SOURCE: https://www.example.com/article
        EXTRACT: First few words ... last few words.

        SOURCE: ...
        EXTRACT: ...

        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        DO NOT MAKE UP YOUR OWN SOURCES; ONLY USE SOURCES YOU FIND FROM A WEB SEARCH.
        """,
    )
    agent = lr.ChatAgent(config)

    agent.enable_message(SeltzSearchTool)

    task = lr.Task(agent, interactive=False)

    user_message = "Can you help me with some questions?"
    task.run(user_message)


if __name__ == "__main__":
    app()
