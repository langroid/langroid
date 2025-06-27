"""
Single-agent to use to chat with the IGVF ArangoDB knowledge-graph (KG) on cloud.

Make sure to set the ARANGODB_PASSWORD in your environment variables.

Run like this (--model is optional, defaults to GPT4o):

python3 examples/kg-chat/chat-arangodb-igvf.py --model litellm/claude-3-5-sonnet-20241022

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

from dotenv import load_dotenv
from fire import Fire
from rich import print

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


def main(
    debug: bool = False,
    model: str = "",
    no_stream: bool = False,
    nocache: bool = False,
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

    url = "https://db.catalog.igvf.org"
    username = "guest"
    db = "igvf"
    pw = os.getenv("ARANGODB_PASSWORD")
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
    Fire(main)
