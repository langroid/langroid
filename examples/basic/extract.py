"""
Extract structured data from text using function_calling/tools.
Based on https://www.kaggle.com/code/thedrcat/using-llms-to-extract-structured-data

"""
import os
import textwrap
import warnings

import typer
from typing import List
from rich import print

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from kaggle_text import kaggle_description
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.vector_store.base import VectorStoreConfig

app = typer.Typer()

setup_colored_logging()


class MethodsList(ToolMessage):
    request: str = "methods_list"
    purpose: str = "list of Machine Learning methods"
    methods: List[str]
    result: str = ""

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                methods=[
                    "XGBoost, bad",
                    "Random Forest, good",
                    "SVM, good",
                ],
                result="",
            ),
        ]


class ExtractorAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)

    def methods_list(self, message: MethodsList) -> str:
        print("Tool handled: Methods list:", message.methods)
        return ",".join(message.methods)


class ExtractorConfig(ChatAgentConfig):
    name = "Extractor"
    debug: bool = False
    max_context_tokens = 500
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = False  # use GPT-4?
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    use_tools = False
    use_functions_api = True
    vecdb: None | VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
    )


def chat(config: ExtractorConfig) -> None:
    print(
        textwrap.dedent(
            """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit
        """
        ).strip()
    )
    agent = ExtractorAgent(config)
    agent.enable_message(
        MethodsList,
        use=True,
        handle=True,
        force=True,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )
    task = Task(
        agent,
        system_message="""
        You are a machine learning engineer analyzing Kaggle competition solutions.
        Your goal is to create a list of Machine Learning methods based on the 
        user's message.
        The methods must be very short names, not long phrases.
        Don't add any methods not mentioned in the solution description.
        Call the methods_list function or Tool to accomplish this.
        """,
        llm_delegate=False,
        single_round=False,
    )
    task.run(kaggle_description)


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    fn_api: bool = typer.Option(False, "--fn_api", "-f", help="use functions api"),
) -> None:
    config = ExtractorConfig(
        use_functions_api=fn_api,
        use_tools=not fn_api,
    )
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat(config)


if __name__ == "__main__":
    app()
