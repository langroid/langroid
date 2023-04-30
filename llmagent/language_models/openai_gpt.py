from llmagent.language_models.base import LanguageModel, LLMConfig
import openai
from dotenv import load_dotenv
import os
import logging

logging.getLogger("openai").setLevel(logging.ERROR)


class OpenAIGPTConfig(LLMConfig):
    type: str = "openai"
    max_tokens: int = 1024
    chat_model: str = "gpt-3.5-turbo"
    completion_model: str = "text-davinci-003"


# Define a class for OpenAI GPT-3 that extends the base class
class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    def __init__(self, config: OpenAIGPTConfig):
        """
        Args:
            config: configuration for openai-gpt model
        """
        super().__init__()
        self.config = config
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _completion_args(self, prompt: str, max_tokens: int):
        return dict(
            model=self.config.completion_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            echo=False,
        )

    def generate(self, prompt: str, max_tokens: int) -> str:
        openai.api_key = self.api_key
        response = openai.Completion.create(**self._completion_args(prompt, max_tokens))
        return response.choices[0].text.strip()

    async def agenerate(self, prompt: str, max_tokens: int) -> str:
        openai.api_key = self.api_key
        response = await openai.Completion.acreate(
            **self._completion_args(prompt, max_tokens)
        )
        return response.choices[0].text.strip()
