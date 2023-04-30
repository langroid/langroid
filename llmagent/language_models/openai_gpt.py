from llmagent.language_models.base import LanguageModel
import openai
from dotenv import load_dotenv
import os
import logging

logging.getLogger("openai").setLevel(logging.ERROR)


# Define a class for OpenAI GPT-3 that extends the base class
class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """

    def __init__(
        self,
        chat_model: str = "gpt-3.5-turbo",
        completion_model: str = "text-davinci-003",
    ):
        """
        Args:
            chat_model: name of chat model
            completion_model: name of completion model
        """
        self.chat_model = chat_model
        self.completion_model = completion_model
        self.max_tokens = 4096
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _completion_args(self, prompt: str, max_tokens: int):
        return dict(
            model=self.completion_model,
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
