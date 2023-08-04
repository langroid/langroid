"""
This is a basic example of a chatbot that uses the GoogleSearchTool:
when the LLM doesn't know the answer to a question, it will use the tool to
search the web for relevant results, and then use the results to answer the
question.
"""
import typer
from rich import print
from rich.prompt import Prompt

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.stateless_tools.google_search_tool import GoogleSearchTool
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()


def chat() -> None:
    print(
        """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit at any point.
        """
    )
    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default="Default: 'You are a helpful assistant'",
    )

    config = ChatAgentConfig(
        system_message=sys_msg,
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
    )
    agent = ChatAgent(config)
    agent.enable_message(GoogleSearchTool)
    task = Task(
        agent,
        system_message="""
        You are a helpful assistant. You will try your best to answer my questions.
        If you don't know you can use up to 5 results from the `web_search` 
        tool/function-call to help you with answering the question.
        Be very concise in your responses, use no more than 1-2 sentences.
        When you answer based on a web search, show me the SOURCE(s) and EXTRACT(s), 
        for example:
        
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        """,
    )
    task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
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
    chat()


if __name__ == "__main__":
    app()
