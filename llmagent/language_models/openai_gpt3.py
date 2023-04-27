from llmagent.language_models.base import LanguageModel
import openai
from dataclasses import dataclass

# Define a class for OpenAI GPT-3 that extends the base class
@dataclass
class OpenAIGPT(LanguageModel):
    """
    Class for OpenAI LLMs
    """
    api_key: str
    chat_model: str = "gpt-3.5-turbo" # model name, not engine name
    completion_model: str = "text-davinci-003"

    def completion_args(self, prompt:str, max_tokens:int):
        return dict(
            model=self.completion_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            echo=False
        )

    def generate(self, prompt:str, max_tokens:int) -> str:
        openai.api_key = self.api_key
        response = openai.Completion.create(**self.completion_args(prompt, max_tokens))
        return response.choices[0].text.strip()

    async def agenerate(self, prompt: str, max_tokens: int) -> str:
        openai.api_key = self.api_key
        response = await openai.Completion.acreate(
            **self.completion_args(prompt, max_tokens)
        )
        return response.choices[0].text.strip()
