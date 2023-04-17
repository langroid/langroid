from abc import ABC, abstractmethod


# Define an abstract base class for language models
class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        pass
