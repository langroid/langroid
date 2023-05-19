from llmagent.utils.logging import setup_colored_logging
from llmagent.utils import configuration
from examples.dockerchat.docker_chat_agent import DockerChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    InformURLMessage,
    FileExistsMessage,
    PythonVersionMessage,
    PythonDependencyMessage,
    ValidateDockerfileMessage,
)
import typer
from llmagent.language_models.base import LLMMessage, Role
from llmagent.agent.base import AgentConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from rich import print

app = typer.Typer()

setup_colored_logging()


class DockerChatAgentConfig(AgentConfig):
    gpt4: bool = False
    debug: bool = False
    cache: bool = True
    stream: bool = True
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
    llm: LLMConfig = LLMConfig(
        type="openai",
        chat_model="gpt-3.5-turbo",
    )
    parsing: ParsingConfig = ParsingConfig(
        chunk_size=100,
        chunk_overlap=10,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


def chat(config: DockerChatAgentConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = "gpt-4"

    print("[blue]Hello I am here to make your dockerfile!")
    print("[cyan]Enter x or q to quit")

    task_messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""
            You are a helpful assistant, able to think step by step.
            """,
        ),
        LLMMessage(
            role=Role.USER,
            content="""
            You are a devops engineer, and you have to write a dockerfile for a python 
            repo. Think step by step about the information you need, to accomplish 
            your task.  If I cannot answer, further breakdown your task into 
            sub-tasks and so on. Start by asking me for the URL of the github repo.
            
            You have access to the following TOOLS to accomplish your task:
            
                        
            """,
        ),
    ]

    agent = DockerChatAgent(config, task_messages)
    agent.enable_message(InformURLMessage)
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    agent.enable_message(ValidateDockerfileMessage)
    agent.enable_message(PythonDependencyMessage)

    agent.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    gpt4: bool = typer.Option(False, "--gpt4", "-4", help="use gpt4"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    config = DockerChatAgentConfig(debug=debug, gpt4=gpt4, cache=not nocache)
    chat(config)


if __name__ == "__main__":
    app()
