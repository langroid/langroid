"""
Variant of chat.py, showing how you can save conversation state, end the script, and
resume the conversation later by re-running the script.

The most basic chatbot example, using the default settings.
A single Agent allows you to chat with a pre-trained Language Model.

Run like this:

python3 examples/basic/chat.py

Use optional arguments to change the settings, e.g.:

-m <local_model_spec>
-ns # no streaming
-d # debug mode
-nc # no cache
-sm <system_message>
-q <initial user msg>

For details on running with local or non-OpenAI models, see:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import logging
import pickle
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global

STATE_CACHE_DIR = ".cache/agent-state"

app = typer.Typer()
logger = logging.getLogger(__name__)
# set the logging level to INFO
logger.setLevel(logging.INFO)
# Create classes for non-OpenAI model configs


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    query: str = typer.Option("", "--query", "-q", help="initial user query or msg"),
    sys_msg: str = typer.Option(
        "You are a helpful assistant. Be concise in your answers.",
        "--sysmsg",
        "-sm",
        help="system message",
    ),
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
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    # use the appropriate config instance depending on model name
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=4096,
        timeout=45,
    )

    # check if history.pkl exists under STATE_CACHE_DIR, and if it does, load it
    # into agent.message_history
    hist_path = Path(STATE_CACHE_DIR) / "history.pkl"
    hist_found = False
    try:
        if hist_path.exists():
            # read the history from the cache
            with open(str(hist_path), "rb") as f:
                msg_history = pickle.load(f)
            n_msgs = len(msg_history)
            logger.info(f"Loaded {n_msgs} messages from cache")
            hist_found = True
        else:
            sys_msg = Prompt.ask(
                "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
                default=sys_msg,
            )

    except Exception:
        logger.warning("Failed to load message history from cache")
        pass

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=llm_config,
    )
    agent = ChatAgent(config)

    if hist_found:
        # overrides sys_msg set in config
        agent.message_history = msg_history

    # use restart=False so the state is not cleared out at start,
    # which allows continuing the conversation.
    task = Task(agent, restart=False)
    # OpenAI models are ok with just a system msg,
    # but in some scenarios, other (e.g. llama) models
    # seem to do better when kicked off with a sys msg and a user msg.
    # In those cases we may want to do task.run("hello") instead.
    if query:
        task.run(query)
    else:
        task.run()

    # Create STATE_CACHE_DIR if it doesn't exist
    Path(STATE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    # Save the conversation state to hist_path
    with open(str(hist_path), "wb") as f:
        pickle.dump(agent.message_history, f)
    logger.info(f"Saved {len(agent.message_history)} messages to cache")


if __name__ == "__main__":
    app()
