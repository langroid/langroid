"""
Meant to be used with local LLMs, using the -l option (see below).
It works fine with GPT4, but does not yet work with the
wizardlm-1.0-uncensored-llama2-13b.Q4_K_M.gguf model:
It fails to follow instructions to send the text to the annotator
using the `recipient_message` too.
It is possbile that bigger models may do better.

2-agent version of chat.py, but now there is a PrivacyAgent that forwards the
 user's text to PrivacyAnnotator, and  checks the work of the PrivacyAnnotator.

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

python3 examples/privacy/chat.py

Use optional arguments to change the settings, e.g.:

-l # use locally running Llama model
-lc 1000 # use local Llama model with context length 1000
    (you typically can set ctx len when spinning up local llm API server)
-ns # no streaming
-d # debug mode
-nc # no cache
-ct momento # use momento cache (instead of redis)

For details on running with local Llama model, see:
https://langroid.github.io/langroid/blog/2023/09/14/using-langroid-with-local-llms/
"""
import typer
from rich import print
from pydantic import BaseSettings
from dotenv import load_dotenv

from examples.privacy.privacy_annotator import PrivacyAnnotator, PrivacyAnnotatorConfig
from examples.privacy.privacy_agent import PrivacyAgent, PrivacyAgentConfig
from langroid.agent.task import Task
from langroid.language_models.base import LocalModelConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()


class CLIOptions(BaseSettings):
    local: bool = False
    api_base: str = "http://localhost:8000/v1"
    local_model: str = ""
    local_ctx: int = 2048
    # use completion endpoint for chat?
    # if so, we should format chat->prompt ourselves, if we know the required syntax
    completion: bool = False

    class Config:
        extra = "forbid"
        env_prefix = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the privacy mask chatbot!
        Enter any text and I will annotate it with sensitive categories and values.
        """
    )

    load_dotenv()

    # create the appropriate OpenAIGPTConfig depending on local model or not

    if opts.local or opts.local_model:
        # assumes local endpoint is either the default http://localhost:8000/v1
        # or if not, it has been set in the .env file as the value of
        # OPENAI_LOCAL.API_BASE
        local_model_config = LocalModelConfig(
            api_base=opts.api_base,
            model=opts.local_model,
            context_length=opts.local_ctx,
            use_completion_for_chat=opts.completion,
        )
        llm_config = OpenAIGPTConfig(
            local=local_model_config,
            timeout=180,
            max_output_tokens=500,
        )
    else:
        # defaults to chat_model = OpenAIChatModel.GPT4
        llm_config = OpenAIGPTConfig()

    annotator_config = PrivacyAnnotatorConfig(
        llm=llm_config,
        vecdb=None,
    )
    annotator_agent = PrivacyAnnotator(annotator_config)
    annotator_task = Task(
        annotator_agent,
        llm_delegate=False,
        single_round=True,
    )

    privacy_config = PrivacyAgentConfig(
        llm=llm_config,
        vecdb=None,
    )
    privacy_agent = PrivacyAgent(privacy_config)
    privacy_task = Task(
        privacy_agent,
        llm_delegate=True,
        single_round=False,
    )
    privacy_task.add_sub_task(annotator_task)

    # local (llama2) models do not like the first message to be empty
    user_message = "Hello." if (opts.local or opts.local_model) else None
    privacy_task.run(user_message)


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    local: bool = typer.Option(False, "--local", "-l", help="use local llm"),
    local_model: str = typer.Option(
        "", "--local_model", "-lm", help="local model path"
    ),
    api_base: str = typer.Option(
        "http://localhost:8000/v1", "--api_base", "-api", help="local model api base"
    ),
    local_ctx: int = typer.Option(
        2048, "--local_ctx", "-lc", help="local llm context size (default 2048)"
    ),
    completion: bool = typer.Option(
        False, "--completion", "-c", help="use completion endpoint for chat"
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
    opts = CLIOptions(
        local=local,
        api_base=api_base,
        local_model=local_model,
        local_ctx=local_ctx,
        completion=completion,
    )
    chat(opts)


if __name__ == "__main__":
    app()
