from abc import ABC, abstractmethod


# Define an abstract base class for language models
class LanguageModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        pass
