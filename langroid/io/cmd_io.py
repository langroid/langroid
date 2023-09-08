from langroid.io.base import InputProvider, OutputProvider
from rich.prompt import Prompt
from rich import print

class CmdInputProvider(InputProvider):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, message, default=""):
        return Prompt.ask("updated" + message, default=default)

class CmdOutputProvider(OutputProvider):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, message: str):
        print("updated" + message)