"""
This is a basic example of a chatbot that uses the GoogleSearchTool:
when the LLM doesn't know the answer to a question, it will use the tool to
search the web for relevant results, and then use the results to answer the
question.

NOTE: running this example requires setting the GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).
"""

import typer
from rich import print
from rich.prompt import Prompt
from pydantic import BaseSettings
from dotenv import load_dotenv

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()

# create classes for other model configs
LiteLLMOllamaConfig = OpenAIGPTConfig.create(prefix="ollama")
litellm_ollama_config = LiteLLMOllamaConfig(
    chat_model="ollama/llama2",
    completion_model="ollama/llama2",
    api_base="http://localhost:11434",
    litellm=True,
    chat_context_length=4096,
    use_completion_for_chat=False,
)
OobaConfig = OpenAIGPTConfig.create(prefix="ooba")
ooba_config = OobaConfig(
    chat_model="local",  # doesn't matter
    completion_model="local",  # doesn't matter
    api_base="http://localhost:8000/v1",  # <- edit if running at a different port
    chat_context_length=2048,
    litellm=False,
    use_completion_for_chat=False,
)


class CLIOptions(BaseSettings):
    model: str = ""

    class Config:
        extra = "forbid"
        env_prefix = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the Google Search chatbot!
        I will try to answer your questions, relying on (summaries of links from) 
        Google Search when needed.
        
        Enter x or q to quit at any point.
        """
    )
    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default="Default: 'You are a helpful assistant'",
    )

    load_dotenv()

    # use the appropriate config instance depending on model name
    if opts.model == "ooba":
        llm_config = ooba_config
    elif opts.model.startswith("ollama"):
        llm_config = litellm_ollama_config
        llm_config.chat_model = opts.model
    else:
        llm_config = OpenAIGPTConfig()

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=llm_config,
        vecdb=None,
    )
    agent = ChatAgent(config)
    agent.enable_message(GoogleSearchTool)
    task = Task(
        agent,
        system_message="""
        You are a helpful assistant. You will try your best to answer my questions.
        If you cannot answer from your own knowledge, you can use up to 5 
        results from the `web_search` tool/function-call to help you with 
        answering the question.
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
    user_message = "Hello." if (opts.model != "") else None
    task.run(user_message)


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
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
    opts = (CLIOptions(model=model),)
    chat(opts)


if __name__ == "__main__":
    app()
