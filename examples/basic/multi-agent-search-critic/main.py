"""
Version of chat-search-assistant.py that is more likely to work local LLMs.

3-Agent system where:
- Assistant takes user's (complex) question, breaks it down into smaller pieces
    if needed
- Searcher takes Assistant's question, uses the Search tool to search the web
    (using DuckDuckGo), and returns a coherent answer to the Assistant.
- Critic takes Assistant's final answer, and provides feedback on it.

Once the Assistant thinks it has enough info to answer the user's question, it
says DONE and presents the answer to the user.

See also: chat-search for a basic single-agent search

Run like this from root of repo:

python3 -m examples.basic.multi-agent-search-critic.main

There are optional args, especially note these:

-m <model_name>: to run with a different LLM model (default: gpt4o)

For example try this question:

did Bach make more music than Beethoven?

You can specify a local LLM in a few different ways, e.g. `-m local/localhost:8000/v1`
or `-m ollama/mistral` etc. See here how to use Langroid with local LLMs:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


"""

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

from langroid.utils.configuration import Settings, set_global

from .assistant_agent import make_assistant_task
from .critic_agent import make_critic_task
from .search_agent import make_search_task

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
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

    assistant_task = make_assistant_task(model)
    search_task = make_search_task(model)
    critic_task = make_critic_task(model)

    assistant_task.add_sub_task([search_task, critic_task])
    question = Prompt.ask("What do you want to know?")
    assistant_task.run(question)


if __name__ == "__main__":
    app()
