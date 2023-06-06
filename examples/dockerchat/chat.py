from llmagent.utils.logging import setup_colored_logging
from llmagent.utils import configuration
from examples.dockerchat.docker_chat_agent import (
    DockerChatAgent,
    PLANNER_INSTRUCTIONS,
    CODE_CHAT_INSTRUCTIONS,
)
from examples.dockerchat.dockerchat_agent_messages import (
    AskURLMessage,
    ValidateDockerfileMessage,
    EntryPointAndCMDMessage,
    RunContainerMessage,
)
import typer
from llmagent.language_models.base import LLMMessage, Role
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.agent.base import AgentConfig
from llmagent.agent.task import Task
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from rich import print

app = typer.Typer()

setup_colored_logging()


class DockerChatAgentConfig(AgentConfig):
    name: str = "DockerExpert"
    gpt4: bool = True
    debug: bool = False
    cache: bool = True
    stream: bool = True
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="llmagent-dockerchat",
        storage_path=".qdrant/llmagent-dockerchat/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
    )
    parsing: ParsingConfig = ParsingConfig(
        chunk_size=100,
    )

    prompts: PromptsConfig = PromptsConfig()


def chat(config: DockerChatAgentConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4

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
            You are a devops engineer, and your goal is to create a working dockerfile 
            to containerize a python repo. Think step by step about the information 
            you need, to accomplish your task, and ask me questions for what you need.  
            If I cannot answer, further refine your question into smaller questions.
            Do not create a dockerfile until you have all the information you need.
            Before showing me a dockerfile, first ask my permission if you can show it.
            Make sure you ask about only ONE item at a time. Do not combine multiple 
            questions into one request.
            Start by asking me for the URL of the github repo.
            """,
        ),
    ]

    agent = DockerChatAgent(config, task_messages)
    # agent.enable_message(RunPython)
    agent.enable_message(AskURLMessage)
    # agent.enable_message(FileExistsMessage)
    # agent.enable_message(PythonVersionMessage)
    agent.enable_message(ValidateDockerfileMessage)
    # agent.enable_message(PythonDependencyMessage)
    agent.enable_message(EntryPointAndCMDMessage)
    agent.enable_message(RunContainerMessage)

    # set up tasks and their hierarchy
    docker_task = Task(
        agent, llm_delegate=True, single_round=False, only_user_quits_root=True
    )
    planner_task = Task(
        agent.planner_agent,
        llm_delegate=True,
        single_round=False,
        system_message=PLANNER_INSTRUCTIONS,
    )
    code_chat_task = Task(
        agent.code_chat_agent,
        llm_delegate=False,
        single_round=True,
        system_message=CODE_CHAT_INSTRUCTIONS,
    )
    docker_task.add_sub_task(planner_task)
    planner_task.add_sub_task(code_chat_task)
    docker_task.run()


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
