from llmagent.utils.logging import setup_colored_logging
from llmagent.utils import configuration
import typer
from llmagent.language_models.base import LLMMessage, Role
from llmagent.agent.base import AgentConfig
from llmagent.agent.cot_agent import COTAgent
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig

import os
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()


class COTAgentConfig(AgentConfig):
    debug: bool = False
    max_tokens: int = 200
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="test",
        storage_path=".qdrant/test/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )
    llm: LLMConfig = LLMConfig(type="openai")
    parsing: ParsingConfig = ParsingConfig(
        chunk_size=100,
        chunk_overlap=10,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


def chat(config: COTAgentConfig) -> None:
    configuration.update_global_settings(config, keys=["debug"])

    print("[blue]Hello I am here to make your dockerfile!")
    print("[cyan]Enter x or q to quit")
    task = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""you are a devops engineer. 
                   You have access to a repo. You have to write a docker file to 
                   containerize it. Come up with a plan to do this, 
                   breaking it down into small steps. 
                   You have to ask me what you need to know in order to complete 
                   your task, ONE STEP AT A TIME. 
                   At any time you can ONLY ASK me a SINGLE QUESTION.
                   I will answer it, and then you will ASK another QUESTION,
                   and son on, until you say you are DONE, and show me the completed 
                   dockerfile.""",
        ),
        LLMMessage(
            role=Role.SYSTEM,
            name="example_user",
            content="Please send me your first QUESTION",
        ),
        LLMMessage(
            role=Role.SYSTEM,
            name="example_assistant",
            content="What language is the repo written in?",
        ),
        LLMMessage(role=Role.SYSTEM, name="example_user", content="Python"),
        LLMMessage(
            role=Role.SYSTEM,
            name="example_assistant",
            content="What version of Python?",
        ),
        LLMMessage(role=Role.SYSTEM, name="example_user", content="3.8"),
        LLMMessage(
            role=Role.USER,
            content="""You are a sophisticated devops engineer, and you 
                           need to write a dockerfile for a repo. Please think 
                           in small steps, and do this gradually, and focus on what 
                           information you need to know to accomplish your task.
                           Please send me your first QUESTION""",
        ),
    ]

    agent = COTAgent(config, task)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )
    agent.start()
    while True:
        print("\n[blue]Human: ", end="")
        msg = input("")
        if msg in ["exit", "quit", "q", "x", "bye"]:
            print("[green] Bye, hope this was useful!")
            break
        agent.respond(msg)


@app.command()
def main(debug: bool = typer.Option(False, "--debug", "-d", help="debug mode")) -> None:
    config = COTAgentConfig(debug=debug)
    chat(config)


if __name__ == "__main__":
    app()
