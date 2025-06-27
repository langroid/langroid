"""
Extract structured data from text using function_calling/tools.
Inspired by this W&B example notebook, but goes beyond, i.e. gets slightly
more structured output to include model quality:
https://wandb.ai/darek/llmapps/reports/Using-LLMs-to-Extract-Structured-Data-OpenAI-Function-Calling-in-Action--Vmlldzo0Nzc0MzQ3

Example usage, to use Langroid tool:
python3 examples/basic/extract.py -nc

Use -f option to use OpenAI function calling API instead of Langroid tool.

"""

import json
import textwrap
from typing import List

import typer
from kaggle_text import kaggle_description
from rich import print

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.pydantic_v1 import BaseModel
from langroid.utils.configuration import Settings, set_global
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


class MethodQuality(BaseModel):
    name: str
    quality: str


class MethodsList(ToolMessage):
    request: str = "methods_list"
    purpose: str = """
        Make a list of Machine Learning methods and their quality
        """
    methods: List[MethodQuality]
    result: str = ""

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                methods=[
                    MethodQuality(name="XGBoost", quality="good"),
                    MethodQuality(name="Random Forest", quality="bad"),
                ],
                result="",
            ),
        ]


class ExtractorAgent(ChatAgent):
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)

    def methods_list(self, message: MethodsList) -> str:
        print(
            f"""
        DONE! Successfully extracted ML Methods list:
        {message.methods}
        """
        )
        return "\n".join(json.dumps(m.dict()) for m in message.methods)


class ExtractorConfig(ChatAgentConfig):
    name = "Extractor"
    debug: bool = False
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = False  # use GPT-4?
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    use_tools = False
    use_functions_api = True
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4o,
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

    task = Task(
        agent,
        system_message="""
        You are a machine learning engineer analyzing Kaggle competition solutions.
        Your goal is to create a list of Machine Learning methods and their 
        quality, based on the user's description. 
        The "quality" can be "good" or "bad", based on your understanding of the 
        description.
        The methods must be very short names, not long phrases.
        Don't add any methods not mentioned in the solution description.
        Call the methods_list function or Tool to accomplish this.
        """,
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
            cache_type="redis",
        )
    )
    chat(config)


if __name__ == "__main__":
    app()
