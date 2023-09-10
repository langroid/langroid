from abc import ABC


class InputProvider(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, message: str, default: str = "") -> str:
        raise NotImplementedError


class OutputProvider(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, message: str, streaming: bool = False) -> None:
        pass

    def flush(self) -> None:
        pass


class IOFactory:
    providers = {}  # type: ignore

    @staticmethod
    def get_provider(name: str):  # type: ignore
        return IOFactory.providers[name]

    @staticmethod
    def set_provider(provider) -> None:  # type: ignore
        IOFactory.providers[provider.name] = provider
