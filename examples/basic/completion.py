# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langroid",
# ]
# ///
"""
Interact with a base completion model, specifically the original GPT-3 base model
(i.e. davinci-002 or babbage-002),
one that has not been instruct-tuned for chat-like conversation.
This uses the legacy OpenAI Completion API.
This API simply takes pure text (NOT dialog) , and returns the LLM's completion.
Note there is no notion of system message here.

Run like this:

python3 examples/basic/completion.py

Use optional arguments to change the settings, e.g.:

-m <local_model_spec>
-ns # no streaming
-d # debug mode
-nc # no cache


For details on running with local or non-OpenAI models, see:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


def multiline_input(prompt_text):
    lines = []
    while True:
        line = Prompt.ask(prompt_text)
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


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
        [blue]Welcome to the basic completion engine.
        Text you enter will be completed by an LLM 
        (Default is a GPT3-class LLM, davinci-002). 
        You can enter multi-line inputs; Enter return TWICE to send your message.
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    # use the appropriate config instance depending on model name
    llm_config = lm.OpenAIGPTConfig(
        completion_model=model or "davinci-002",  # or "babbage-002"
        chat_context_length=4096,
        timeout=45,
        use_chat_for_completion=False,
    )
    llm = lm.OpenAIGPT(llm_config)

    print()
    while True:
        print("\n")
        user_msg = multiline_input("[blue]You[/blue]")
        if user_msg.lower() in ["q", "x"]:
            break
        print("\nBot: ")
        response = llm.generate(prompt=user_msg, max_tokens=50)

        if response.cached:
            print(f"[red](Cached)[/red] [green] {response.message}[/green]")


if __name__ == "__main__":
    app()
