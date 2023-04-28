from abc import ABC, abstractmethod
from typing import List, Tuple
from llmagent.mytypes import Document
from llmagent.utils.configuration import settings
from llmagent.utils.output.printing import show_if_debug
from llmagent.language_models.config import LLMConfig
from llmagent.prompts.templates import EXTRACTION_PROMPT, SUMMARY_ANSWER_PROMPT
from llmagent.prompts.dialog import collate_chat_history
import aiohttp
import asyncio



# Define an abstract base class for language models
class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """

    @staticmethod
    def create(config: LLMConfig):
        """
        Create a language model.
        Args:
            config: configuration for language model
        Returns: instance of language model
        """
        from llmagent.language_models.openai_gpt import OpenAIGPT
        cls = dict(
            openai=OpenAIGPT,
        ).get(config.type, OpenAIGPT)
        return cls()

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        pass

    def __call__(self, prompt: str, max_tokens: int) -> str:
        return self.generate(prompt, max_tokens)

    def followup_to_standalone(
            self,
            chat_history: List[Tuple[str]],
            question:str
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
        show_if_debug(prompt, "FOLLOWUP->STANDALONE-PROMPT= ")
        standalone = self.generate(prompt=prompt, max_tokens=1024).strip()
        show_if_debug(prompt, "FOLLOWUP->STANDALONE-RESPONSE= ")
        return standalone

    async def get_verbatim_extract_async(self, question: str, passage: Document) -> str:
        """
        Asynchronously, get verbatim extract from passage
        that is relevant to a question.
        Asynch allows parallel calls to the LLM API.
        """
        async with aiohttp.ClientSession():
            templatized_prompt = EXTRACTION_PROMPT
            show_if_debug(EXTRACTION_PROMPT, "EXTRACT-PROMPT= ")
            final_prompt = templatized_prompt.format(question=question, content=passage)
            final_extract = await self.agenerate(prompt=final_prompt,
                                                    max_tokens=1024)
            show_if_debug(final_extract, "EXTRACT-RESPONSE= ")
        return final_extract.strip()


    async def _get_verbatim_extracts(
            self,
            question: str,
            passages: List[Document],
    ) -> List[str]:
        async with aiohttp.ClientSession():
            verbatim_extracts = await asyncio.gather(
                *(self.get_verbatim_extract_async(question, P) for P in passages)
            )
        metadatas = [P.metadata for P in passages]
        # return with metadata so we can use it downstream, e.g. to cite sources
        return [Document(content=e, metadata=m) for
                e, m in zip(verbatim_extracts, metadatas)]

    def get_verbatim_extracts(
            self,
            question: str,
            passages: List[Document]
    ) -> List[Document]:
        """
        From each passage, extract verbatim text that is relevant to a question,
        using concurrent API calls to the LLM.
        Args:
            question: question to be answered
            passages: list of passages from which to extract relevant verbatim text
            LLM: LanguageModel to use for generating the prompt and extract
        Returns:
            list of verbatim extracts from passages that are relevant to question
        """
        docs = asyncio.run(self._get_verbatim_extracts(question, passages))
        return docs

    def get_summary_answer(self, question: str, passages: List[Document]) -> Document:
        """
        Given a question and a list of (possibly) doc snippets,
        generate an answer if possible
        Args:
            question: question to answer
            passages: list of `Document` objects each containing a possibly relevant
                snippet, and metadata
        Returns:
            a `Document` object containing the answer,
            and metadata containing source citations

        """

        # Define an auxiliary function to transform the list of passages into a single string
        def stringify_passages(passages):
            return "\n".join([
                f"""
                Extract: {p.content}
                Source: {p.metadata["source"]}
                """
                for p in passages])

        passages = stringify_passages(passages)
        # Substitute Q and P into the templatized prompt
        final_prompt = SUMMARY_ANSWER_PROMPT.format(
            question=f"Question:{question}",
            extracts=passages
        )
        show_if_debug(final_prompt, "SUMMARIZE_PROMPT= ")
        # Generate the final verbatim extract based on the final prompt
        final_answer = self.generate(prompt=final_prompt, max_tokens=1024).strip()
        show_if_debug(final_answer, "SUMMARIZE_RESPONSE= ")
        parts = final_answer.split("SOURCE:", maxsplit=1)
        if len(parts) > 1:
            content = parts[0].strip()
            sources = parts[1].strip()
        else:
            content = final_answer
            sources = ""
        return Document(content=content, metadata={"source": "SOURCE: " + sources})

