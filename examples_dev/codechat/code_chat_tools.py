import logging
from typing import List

from langroid.agent.base import ToolMessage

logger = logging.getLogger(__name__)


class ShowFileContentsMessage(ToolMessage):
    request: str = "show_file_contents"
    # name should exactly match method  name in agent
    # below will be fields that will be used by the agent method to handle the message.
    purpose: str = "To see the contents of a file path <filepath> in the repo."
    filepath: str = None  # must provide or will raise error
    result: str = ""

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                filepath="src/main.py",
                result="import os\n\nprint('hello world')\n",
            ),
        ]


class ShowDirContentsMessage(ToolMessage):
    request: str = "show_dir_contents"
    # name should exactly match method  name in agent
    # below will be fields that will be used by the agent method to handle the message.
    purpose: str = "To see the contents of a directory path <dirpath> in the repo."
    dirpath: str = "src/"
    result: str = ""

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                dirpath="src/",
                result="main.py\napp.py\n",
            ),
        ]
