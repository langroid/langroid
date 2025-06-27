"""
Meant to be used with local LLMs, using the -m option (see below).

You type a sentence containing potentially sensitive information, and the agent
will annotate sensitive portions of the sentence with the appropriate category.
You can configure PrivacyAnnotator to recognize only specific sensitive
categories, currently defaults to: ["Medical", "CreditCard", "SSN", "Name"]

Example input:
    "John is 45 years old, lives in Ohio, makes 45K a year, and has diabetes."

Example output:
    "[Name: John] is 45 years old, lives in Ohio, makes 45K a year,
        and has [Medical: diabetes]."

Run like this:

python3 examples/privacy/annotate.py

Use optional arguments to change the settings, e.g.:

-m ollama/mistral:latest # use locally LLM
-d # debug mode
-nc # no cache

For details on running with local LLMs, see here:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
"""

import typer
from dotenv import load_dotenv
from rich import print

import langroid as lr
import langroid.language_models as lm
from examples.privacy.privacy_annotator import PrivacyAnnotator, PrivacyAnnotatorConfig
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


# create classes for other model configs


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
        [blue]Welcome to the privacy mask chatbot!
        Enter any text and I will annotate it with sensitive categories and values.
        """
    )

    load_dotenv()
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=8000,  # adjust based on model
        timeout=90,
    )

    config = PrivacyAnnotatorConfig(
        llm=llm_config,
        vecdb=None,
    )
    agent = PrivacyAnnotator(config)
    task = lr.Task(agent)
    # local (llama2) models do not like the first message to be empty
    user_message = "Hello." if (model != "") else None
    task.run(user_message)


if __name__ == "__main__":
    app()
