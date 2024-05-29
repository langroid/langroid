from langroid.language_models.base import LanguageModel as LanguageModel
from langroid.mytypes import Document as Document
from langroid.prompts.dialog import collate_chat_history as collate_chat_history
from langroid.prompts.templates import EXTRACTION_PROMPT as EXTRACTION_PROMPT

async def get_verbatim_extract_async(
    question: str, passage: Document, LLM: LanguageModel
) -> str: ...
def get_verbatim_extracts(
    question: str, passages: list[Document], LLM: LanguageModel
) -> list[Document]: ...
def followup_to_standalone(
    LLM: LanguageModel, chat_history: list[tuple[str, str]], question: str
) -> str: ...
