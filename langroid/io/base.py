from abc import ABC

class InputProvider(ABC):
    def __init__(self, name):
        self.name = name

    def __call__(self, message, default=""):
        pass

class OutputProvider(ABC):
    def __init__(self, name):
        self.name = name

    def __call__(self, message: str):
        pass

class IOFactory:
    providers = {}

    @staticmethod
    def get_provider(name):
        return IOFactory.providers[name]

    @staticmethod
    def set_provider(provider):
        IOFactory.providers[provider.name] = provider