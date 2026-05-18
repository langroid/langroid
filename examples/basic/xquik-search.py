"""
This is a basic example of a chatbot that uses Xquik to search public X posts.

Run like this:

    python3 examples/basic/xquik-search.py

or

    uv run examples/basic/xquik-search.py -m groq/deepseek-r1-distill-llama-70b

Set XQUIK_API_KEY in your `.env` file before running this example.
"""

import typer
from dotenv import load_dotenv
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.xquik_search_tool import XquikSearchTool
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
        [blue]Welcome to the Xquik X Search chatbot!
        Ask about recent public X posts, accounts, hashtags, or X search operators.

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

    search_tool_handler_method = XquikSearchTool.name()
    config = lr.ChatAgentConfig(
        name="XquikSeeker",
        handle_llm_no_tool="user",
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a helpful assistant. Use `{search_tool_handler_method}` when
        the user asks about public X posts, recent posts, hashtags, accounts,
        or X search operators. The tool query may include operators such as
        from:user, #hashtags, exact phrases, since:YYYY-MM-DD, until:YYYY-MM-DD,
        and min_faves:N.

        Wait for the tool result before answering. Do not invent posts, authors,
        URLs, or metrics.

        Keep answers concise. When using search results, cite the post URLs
        returned by the tool.
        """,
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(XquikSearchTool)

    task = lr.Task(agent, interactive=False)
    task.run('Search X for recent posts from xquikcom that mention "API".')


if __name__ == "__main__":
    app()
