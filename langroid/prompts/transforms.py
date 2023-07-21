import asyncio
from typing import List, Tuple

import aiohttp

from langroid.language_models.base import LanguageModel
from langroid.mytypes import Document
from langroid.prompts.dialog import collate_chat_history
from langroid.prompts.templates import EXTRACTION_PROMPT


async def get_verbatim_extract_async(
    question: str,
    passage: Document,
    LLM: LanguageModel,
) -> str:
    """
    Asynchronously, get verbatim extract from passage that is relevant to a question.
    """
    async with aiohttp.ClientSession():
        templatized_prompt = EXTRACTION_PROMPT
        final_prompt = templatized_prompt.format(question=question, content=passage)
        final_extract = await LLM.agenerate(prompt=final_prompt, max_tokens=1024)

    return final_extract.message.strip()


async def _get_verbatim_extracts(
    question: str,
    passages: List[Document],
    LLM: LanguageModel,
) -> List[Document]:
    async with aiohttp.ClientSession():
        verbatim_extracts = await asyncio.gather(
            *(get_verbatim_extract_async(question, P, LLM) for P in passages)
        )
    metadatas = [P.metadata for P in passages]
    # return with metadata so we can use it downstream, e.g. to cite sources
    return [
        Document(content=e, metadata=m) for e, m in zip(verbatim_extracts, metadatas)
    ]


def get_verbatim_extracts(
    question: str,
    passages: List[Document],
    LLM: LanguageModel,
) -> List[Document]:
    """
    From each passage, extract verbatim text that is relevant to a question,
    using concurrent API calls to the LLM.
    Args:
        question: question to be answered
        passages: list of passages from which to extract relevant verbatim text
        LLM: LanguageModel to use for generating the prompt and extract
    Returns:
        list of verbatim extracts (Documents) from passages that are relevant to
        question
    """
    return asyncio.run(_get_verbatim_extracts(question, passages, LLM))


def followup_to_standalone(
    LLM: LanguageModel, chat_history: List[Tuple[str, str]], question: str
) -> str:
    """
    Given a chat history and a question, convert it to a standalone question.
    Args:
        chat_history: list of tuples of (question, answer)
        query: follow-up question

    Returns: standalone version of the question
    """
    history = collate_chat_history(chat_history)

    prompt = f"""
    Given the conversationn below, and a follow-up question, rephrase the follow-up 
    question as a standalone question.
    
    Chat history: {history}
    Follow-up question: {question} 
    """.strip()
    standalone = LLM.generate(prompt=prompt, max_tokens=1024).message.strip()
    return standalone
