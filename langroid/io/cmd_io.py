from rich import print
from rich.prompt import Prompt

from langroid.io.base import InputProvider, OutputProvider


class CmdInputProvider(InputProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, message: str, default: str = "") -> str:
        return Prompt.ask("updated" + message, default=default)


class CmdOutputProvider(OutputProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(self, message: str) -> None:
        print("updated" + message)
