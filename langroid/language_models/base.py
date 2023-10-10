import asyncio
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import aiohttp
from pydantic import BaseModel, BaseSettings

from langroid.cachedb.momento_cachedb import MomentoCacheConfig
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.config import Llama2FormatterConfig, PromptFormatterConfig
from langroid.mytypes import Document
from langroid.parsing.agent_chats import parse_message
from langroid.parsing.json import top_level_json_field
from langroid.prompts.dialog import collate_chat_history
from langroid.prompts.templates import (
    EXTRACTION_PROMPT_GPT4,
    SUMMARY_ANSWER_PROMPT_GPT4,
)
from langroid.utils.configuration import settings
from langroid.utils.output.printing import show_if_debug

logger = logging.getLogger(__name__)


class LLMConfig(BaseSettings):
    type: str = "openai"
    formatter: None | PromptFormatterConfig = Llama2FormatterConfig()
    timeout: int = 20  # timeout for API requests
    chat_model: str = ""
    completion_model: str = ""
    temperature: float = 0.0
    chat_context_length: int = 1024
    completion_context_length: int = 1024
    max_output_tokens: int = 1024  # generate at most this many tokens
    # if input length + max_output_tokens > context length of model,
    # we will try shortening requested output
    min_output_tokens: int = 64
    use_completion_for_chat: bool = False  # use completion model for chat?
    # use chat model for completion? For OpenAI models, this MUST be set to True!
    use_chat_for_completion: bool = True
    stream: bool = True  # stream output from API?
    cache_config: None | RedisCacheConfig | MomentoCacheConfig = None

    # Dict of model -> (input/prompt cost, output/completion cost)
    chat_cost_per_1k_tokens: Tuple[float, float] = (0.0, 0.0)
    completion_cost_per_1k_tokens: Tuple[float, float] = (0.0, 0.0)


class LLMFunctionCall(BaseModel):
    """
    Structure of LLM response indicate it "wants" to call a function.
    Modeled after OpenAI spec for `function_call` field in ChatCompletion API.
    """

    name: str  # name of function to call
    to: str = ""  # intended recipient
    arguments: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return "FUNC: " + json.dumps(self.dict(), indent=2)


class LLMFunctionSpec(BaseModel):
    """
    Description of a function available for the LLM to use.
    To be used when calling the LLM `chat()` method with the `functions` parameter.
    Modeled after OpenAI spec for `functions` fields in ChatCompletion API.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


class LLMTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Role(str, Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class LLMMessage(BaseModel):
    """
    Class representing message sent to, or received from, LLM.
    """

    role: Role
    name: Optional[str] = None
    content: str
    function_call: Optional[LLMFunctionCall] = None

    def api_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for API request.
        Returns:
            dict: dictionary representation of LLM message
        """
        d = self.dict()
        # drop None values since API doesn't accept them
        dict_no_none = {k: v for k, v in d.items() if v is not None}
        if "name" in dict_no_none and dict_no_none["name"] == "":
            # OpenAI API does not like empty name
            del dict_no_none["name"]
        if "function_call" in dict_no_none:
            # arguments must be a string
            if "arguments" in dict_no_none["function_call"]:
                dict_no_none["function_call"]["arguments"] = json.dumps(
                    dict_no_none["function_call"]["arguments"]
                )
        return dict_no_none

    def __str__(self) -> str:
        if self.function_call is not None:
            content = "FUNC: " + json.dumps(self.function_call)
        else:
            content = self.content
        name_str = f" ({self.name})" if self.name else ""
        return f"{self.role} {name_str}: {content}"


class LLMResponse(BaseModel):
    """
    Class representing response from LLM.
    """

    message: str
    function_call: Optional[LLMFunctionCall] = None
    usage: Optional[LLMTokenUsage]
    cached: bool = False

    def to_LLMMessage(self) -> LLMMessage:
        content = self.message
        role = Role.ASSISTANT if self.function_call is None else Role.FUNCTION
        name = None if self.function_call is None else self.function_call.name
        return LLMMessage(
            role=role,
            content=content,
            name=name,
            function_call=self.function_call,
        )

    def get_recipient_and_message(
        self,
    ) -> Tuple[str, str]:
        """
        If `message` or `function_call` of an LLM response contains an explicit
        recipient name, return this recipient name and `message` stripped
        of the recipient name if specified.

        Two cases:
        (a) `message` contains "TO: <name> <content>", or
        (b) `message` is empty and `function_call` with `to: <name>`

        Returns:
            (str): name of recipient, which may be empty string if no recipient
            (str): content of message

        """

        if self.function_call is not None:
            # in this case we ignore message, since all information is in function_call
            msg = ""
            # recipient may either have been specified as a special field "to" in
            # function_call, or as a parameter "recipient" in the arguments
            # (the latter can happen when using a Tool that has a 'recipient' parameter)
            recipient = self.function_call.to
            if recipient == "":
                args = self.function_call.arguments
                if isinstance(args, dict):
                    recipient = args.get("recipient", "")
            return recipient, msg
        else:
            msg = self.message

        # It's not a function call, so continue looking to see
        # if a recipient is specified in the message.

        # First check if message contains "TO: <recipient> <content>"
        recipient_name, content = parse_message(msg) if msg is not None else ("", "")
        # check if there is a top level json that specifies 'recipient',
        # and retain the entire message as content.
        if recipient_name == "":
            recipient_name = top_level_json_field(msg, "recipient") if msg else ""
            content = msg
        return recipient_name, content


# Define an abstract base class for language models
class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @staticmethod
    def create(config: Optional[LLMConfig]) -> Optional["LanguageModel"]:
        """
        Create a language model.
        Args:
            config: configuration for language model
        Returns: instance of language model
        """
        from langroid.language_models.azure_openai import AzureGPT
        from langroid.language_models.openai_gpt import OpenAIGPT

        if config is None or config.type is None:
            return None

        openai: Union[Type[AzureGPT], Type[OpenAIGPT]]

        if config.type == "azure":
            openai = AzureGPT
        else:
            openai = OpenAIGPT
        cls = dict(
            openai=openai,
        ).get(config.type, openai)
        return cls(config)  # type: ignore

    @staticmethod
    def user_assistant_pairs(lst: List[str]) -> List[Tuple[str, str]]:
        """
        Given an even-length sequence of strings, split into a sequence of pairs

        Args:
            lst (List[str]): sequence of strings

        Returns:
            List[Tuple[str,str]]: sequence of pairs of strings
        """
        evens = lst[::2]
        odds = lst[1::2]
        return list(zip(evens, odds))

    @staticmethod
    def get_chat_history_components(
        messages: List[LLMMessage],
    ) -> Tuple[str, List[Tuple[str, str]], str]:
        """
        From the chat history, extract system prompt, user-assistant turns, and
        final user msg.

        Args:
            messages (List[LLMMessage]): List of messages in the chat history

        Returns:
            Tuple[str, List[Tuple[str,str]], str]:
                system prompt, user-assistant turns, final user msg

        """
        # Handle various degenerate cases
        messages = [m for m in messages]  # copy
        DUMMY_SYS_PROMPT = "You are a helpful assistant."
        DUMMY_USER_PROMPT = "Follow the instructions above."
        if len(messages) == 0 or messages[0].role != Role.SYSTEM:
            logger.warning("No system msg, creating dummy system prompt")
            messages.insert(0, LLMMessage(content=DUMMY_SYS_PROMPT, role=Role.SYSTEM))
        system_prompt = messages[0].content

        # now we have messages = [Sys,...]
        if len(messages) == 1:
            logger.warning(
                "Got only system message in chat history, creating dummy user prompt"
            )
            messages.append(LLMMessage(content=DUMMY_USER_PROMPT, role=Role.USER))

        # now we have messages = [Sys, msg, ...]

        if messages[1].role != Role.USER:
            messages.insert(1, LLMMessage(content=DUMMY_USER_PROMPT, role=Role.USER))

        # now we have messages = [Sys, user, ...]
        if messages[-1].role != Role.USER:
            logger.warning(
                "Last message in chat history is not a user message,"
                " creating dummy user prompt"
            )
            messages.append(LLMMessage(content=DUMMY_USER_PROMPT, role=Role.USER))

        # now we have messages = [Sys, user, ..., user]
        # so we omit the first and last elements and make pairs of user-asst messages
        conversation = [m.content for m in messages[1:-1]]
        user_prompt = messages[-1].content
        pairs = LanguageModel.user_assistant_pairs(conversation)
        return system_prompt, pairs, user_prompt

    @abstractmethod
    def set_stream(self, stream: bool) -> bool:
        """Enable or disable streaming output from API.
        Return previous value of stream."""
        pass

    @abstractmethod
    def get_stream(self) -> bool:
        """Get streaming status"""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> LLMResponse:
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, max_tokens: int) -> LLMResponse:
        pass

    @abstractmethod
    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int,
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
    ) -> LLMResponse:
        pass

    def __call__(self, prompt: str, max_tokens: int) -> LLMResponse:
        return self.generate(prompt, max_tokens)

    def chat_context_length(self) -> int:
        return self.config.chat_context_length

    def completion_context_length(self) -> int:
        return self.config.completion_context_length

    def chat_cost(self) -> Tuple[float, float]:
        return self.config.chat_cost_per_1k_tokens

    def followup_to_standalone(
        self, chat_history: List[Tuple[str, str]], question: str
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
        standalone = self.generate(prompt=prompt, max_tokens=1024).message.strip()
        show_if_debug(prompt, "FOLLOWUP->STANDALONE-RESPONSE= ")
        return standalone

    async def get_verbatim_extract_async(self, question: str, passage: Document) -> str:
        """
        Asynchronously, get verbatim extract from passage
        that is relevant to a question.
        Asynch allows parallel calls to the LLM API.
        """
        async with aiohttp.ClientSession():
            templatized_prompt = EXTRACTION_PROMPT_GPT4
            final_prompt = templatized_prompt.format(
                question=question, content=passage.content
            )
            show_if_debug(final_prompt, "EXTRACT-PROMPT= ")
            final_extract = await self.agenerate(prompt=final_prompt, max_tokens=1024)
            show_if_debug(final_extract.message.strip(), "EXTRACT-RESPONSE= ")
        return final_extract.message.strip()

    async def _get_verbatim_extracts(
        self,
        question: str,
        passages: List[Document],
    ) -> List[Document]:
        async with aiohttp.ClientSession():
            verbatim_extracts = await asyncio.gather(
                *(self.get_verbatim_extract_async(question, P) for P in passages)
            )
        metadatas = [P.metadata for P in passages]
        # return with metadata so we can use it downstream, e.g. to cite sources
        return [
            Document(content=e, metadata=m)
            for e, m in zip(verbatim_extracts, metadatas)
        ]

    def get_verbatim_extracts(
        self, question: str, passages: List[Document]
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

        # Define an auxiliary function to transform the list of
        # passages into a single string
        def stringify_passages(passages: List[Document]) -> str:
            return "\n".join(
                [
                    f"""
                Extract: {p.content}
                Source: {p.metadata.source}
                """
                    for p in passages
                ]
            )

        passages_str = stringify_passages(passages)
        # Substitute Q and P into the templatized prompt

        final_prompt = SUMMARY_ANSWER_PROMPT_GPT4.format(
            question=f"Question:{question}", extracts=passages_str
        )
        show_if_debug(final_prompt, "SUMMARIZE_PROMPT= ")
        # Generate the final verbatim extract based on the final prompt
        llm_response = self.generate(prompt=final_prompt, max_tokens=1024)
        final_answer = llm_response.message.strip()
        show_if_debug(final_answer, "SUMMARIZE_RESPONSE= ")
        parts = final_answer.split("SOURCE:", maxsplit=1)
        if len(parts) > 1:
            content = parts[0].strip()
            sources = parts[1].strip()
        else:
            content = final_answer
            sources = ""
        return Document(
            content=content,
            metadata={
                "source": "SOURCE: " + sources,
                "cached": llm_response.cached,
            },
        )


class StreamingIfAllowed:
    """Context to temporarily enable or disable streaming, if allowed globally via
    `settings.stream`"""

    def __init__(self, llm: LanguageModel, stream: bool = True):
        self.llm = llm
        self.stream = stream

    def __enter__(self) -> None:
        self.old_stream = self.llm.set_stream(settings.stream and self.stream)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.llm.set_stream(self.old_stream)
