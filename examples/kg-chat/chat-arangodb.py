"""
Single-agent to use to chat with an existing ArangoDB knowledge-graph (KG) on cloud,
or locally.
If you have an existing ArangoDB instance, you can
chat with it by specifying its URL, username, password, and database name in the dialog.

Run like this (--model is optional, defaults to GPT4o):

python3 examples/kg-chat/chat-arangodb.py --model litellm/claude-3-5-sonnet-20241022

If using litellm, remember to install langroid with the litellm extra, e.g.
pip install "langroid[litellm]"

See these guides for info on setting up langroid to use Open/Local LLMs
and other non-OpenAI LLMs:
- https://langroid.github.io/langroid/tutorials/local-llm-setup/
- https://langroid.github.io/langroid/tutorials/non-openai-llms/
"""

import logging
import os
from typing import Optional

import typer
from adb_cloud_connector import get_temp_credentials
from arango.client import ArangoClient
from arango_datasets import Datasets
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.prompt import Prompt

import langroid.language_models as lm
from langroid import TaskConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.special.arangodb.arangodb_agent import (
    ArangoChatAgent,
    ArangoChatAgentConfig,
    ArangoSettings,
)
from langroid.agent.task import Task
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import SEND_TO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Add this
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer()


class MyArangoChatAgent(ArangoChatAgent):
    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        response = super().user_response(msg)
        if response is not None and response.content == "r":

            self.clear_history(1)  # remove all msgs after system msg
            n_msgs = len(self.message_history)
            assert n_msgs == 1
            logger.warning("Reset Agent history, only system msg remains")
            # prompt user again
            return super().user_response(msg)

        return response


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=nocache,
            stream=not no_stream,
        )
    )
    print(
        """
        [blue]Welcome to ArangoDB Knowledge Graph RAG chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    url = Prompt.ask(
        "ArangoDB URL (enter 'got' for Game of Thrones dataset) ",
        default="https://db.catalog.igvf.org",
    )
    username = Prompt.ask(
        "ArangoDB username ",
        default="guest",
    )
    db = Prompt.ask(
        "ArangoDB database ",
        default="igvf",
    )
    pw = Prompt.ask(
        "ArangoDB password ",
        default="",
    )
    pw = pw or os.getenv("ARANGODB_PASSWORD")
    if url == "got":
        print(
            """
            No URL supplied, using Game of Thrones dataset from cloud, see here:
            https://docs.arangodb.com/3.11/components/tools/arango-datasets/
            """
        )
        connection = get_temp_credentials(tutorialName="langroid")
        client = ArangoClient(hosts=connection["url"])

        db = client.db(
            connection["dbName"],
            connection["username"],
            connection["password"],
            verify=True,
        )
        datasets = Datasets(db)
        ArangoChatAgent.cleanup_graph_db(db)
        assert len(datasets.list_datasets()) > 0, "No datasets found"

        DATASET = "GAME_OF_THRONES"  # a small dataset
        info = datasets.dataset_info(DATASET)
        assert info["label"] == DATASET
        datasets.load(DATASET, batch_size=100, preserve_existing=False)
        arango_settings = ArangoSettings(db=db, client=client)
    else:
        arango_settings = ArangoSettings(
            url=url,
            username=username,
            database=db,
            password=pw,
        )

    arango_agent = MyArangoChatAgent(
        ArangoChatAgentConfig(
            name="Arango",
            chat_mode=True,
            arango_settings=arango_settings,
            prepopulate_schema=True,
            use_functions_api=False,
            use_tools=True,
            database_created=True,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4o,
                chat_context_length=128_000,
            ),
            human_prompt=(
                "Human (respond, or x/q to quit, r to reset history, "
                "or hit enter to continue)"
            ),
        )
    )

    task_config = TaskConfig(addressing_prefix=SEND_TO)
    arango_task = Task(
        arango_agent,
        # user not awaited, UNLESS LLM explicitly addresses user via recipient_tool
        interactive=False,
        config=task_config,
    )

    arango_task.run(
        "Can you help with some queries? "
        "Be concise and ask me for clarifications when you're not sure what I mean."
    )

    # The above runs the app in a continuous chat.
    # Alternatively, to set up a task to answer a single query and quit when done:

    # set up arango_agent above with chat_mode=False, set up arango_task as above,
    # then run the task with a single query, e.g.:

    # result = arango_task.run("What is the location of the gene BRCA1?")

    # You can have this in a loop with the user, like so:

    # while True:
    #     query = Prompt.ask("Enter your query")
    #     if query in ["x", "q"]:
    #         break
    #     result = arango_task.run(query)
    #     print(result.content)


if __name__ == "__main__":
    app()
