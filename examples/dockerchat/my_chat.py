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
from rich import print
import textwrap

import ask_user as au
import process_repo as pr
import construct_llm_message as llmmsg

app = typer.Typer()

setup_colored_logging()


class COTAgentConfig(AgentConfig):
    debug: bool = False
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
    llm: LLMConfig = LLMConfig(type="openai")
    parsing: ParsingConfig = ParsingConfig(
        chunk_size=100,
        chunk_overlap=10,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


def chat(config: COTAgentConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream"])

    print("[blue]Hello I am here to make your dockerfile!")
    task = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""You are a devops engineer. For a given repo, you have to 
            write a dockerfile to containerize it. I'll provide a short summary about my project. 
            DON"T make any assumption, ask me what you need to know in order to complete your task and show 
            me the completed dockerfile.""",
        ),
    ]

    print("\n[blue]Please specify the URL to your repo: ", end="")
    repo_url = input("")

    # get repo metadata
    exit_code, repo_metadata = pr.extract_repo_metadata(repo_url)

    if exit_code == 200:
        print("Success!")
        print(repo_metadata)
    else:
        print(f"Extraction faild exit code: {exit_code}")
        exit(1)

    # get some info from the user
    entry_cmd = au.get_entry_startup_cmd()
    port = au.get_expose_port()

    # construct a new LLMMessage based on the provided inputs and extracted data
    new_task = llmmsg.construct_LLMMEssage(repo_metadata, port, entry_cmd)
    task.append(new_task)

    agent = COTAgent(config, task)
    agent.llm.set_stream(config.stream)
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
