"""
This example shows how to use Langroid to interact directly with an OpenAI GPT chat model,
i.e., without wrapping it in an Agent.

Run as follows:

python3 examples/quick-start/try-llm.py

For more explanation see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/llm-interaction/)
"""

import typer
from rich import print
from rich.prompt import Prompt

import langroid as lr

Role = lr.language_models.Role
LLMMessage = lr.language_models.LLMMessage

app = typer.Typer()


def chat() -> None:
    print("[blue]Welcome to langroid!")

    cfg = lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o,
    )

    mdl = lr.language_models.OpenAIGPT(cfg)
    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assitant"),
    ]
    while True:
        message = Prompt.ask("[blue]Human")
        if message in ["x", "q"]:
            print("[magenta]Bye!")
            break
        messages.append(LLMMessage(role=Role.USER, content=message))

        # use the OpenAI ChatCompletion API to generate a response
        response = mdl.chat(messages=messages, max_tokens=200)

        messages.append(response.to_LLMMessage())

        print("[green]Bot: " + response.message)


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()
