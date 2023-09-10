import sys

from rich import print
from rich.prompt import Prompt

from langroid.io.base import InputProvider, OutputProvider
from langroid.utils.constants import Colors


class CmdInputProvider(InputProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, message: str, default: str = "") -> str:
        return Prompt.ask(message, default=default)


class CmdOutputProvider(OutputProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, message: str, streaming: bool = False) -> None:
        if streaming:
            sys.stdout.write(Colors().GREEN + message)
            sys.stdout.flush()
        else:
            print(message)

    def flush(self) -> None:
        pass
