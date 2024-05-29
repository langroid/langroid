from abc import ABC
from typing import Any, Callable, Coroutine

from _typeshed import Incomplete
from pydantic import BaseSettings, ValidationError

from langroid.agent.chat_document import (
    ChatDocMetaData as ChatDocMetaData,
)
from langroid.agent.chat_document import (
    ChatDocument as ChatDocument,
)
from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.base import (
    LanguageModel as LanguageModel,
)
from langroid.language_models.base import (
    LLMConfig as LLMConfig,
)
from langroid.language_models.base import (
    LLMMessage as LLMMessage,
)
from langroid.language_models.base import (
    LLMResponse as LLMResponse,
)
from langroid.language_models.base import (
    LLMTokenUsage as LLMTokenUsage,
)
from langroid.language_models.base import (
    StreamingIfAllowed as StreamingIfAllowed,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPT as OpenAIGPT,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPTConfig as OpenAIGPTConfig,
)
from langroid.mytypes import Entity as Entity
from langroid.parsing.parse_json import extract_top_level_json as extract_top_level_json
from langroid.parsing.parser import Parser as Parser
from langroid.parsing.parser import ParsingConfig as ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig as PromptsConfig
from langroid.utils.configuration import settings as settings
from langroid.utils.constants import NO_ANSWER as NO_ANSWER
from langroid.utils.output import status as status
from langroid.vector_store.base import (
    VectorStore as VectorStore,
)
from langroid.vector_store.base import (
    VectorStoreConfig as VectorStoreConfig,
)

console: Incomplete
logger: Incomplete

class AgentConfig(BaseSettings):
    name: str
    debug: bool
    vecdb: VectorStoreConfig | None
    llm: LLMConfig | None
    parsing: ParsingConfig | None
    prompts: PromptsConfig | None
    show_stats: bool
    def check_name_alphanum(cls, v: str) -> str: ...

def noop_fn(*args: list[Any], **kwargs: dict[str, Any]) -> None: ...

class Agent(ABC):
    config: Incomplete
    lock: Incomplete
    dialog: Incomplete
    llm_tools_map: Incomplete
    llm_tools_handled: Incomplete
    llm_tools_usable: Incomplete
    interactive: Incomplete
    total_llm_token_cost: float
    total_llm_token_usage: int
    token_stats_str: str
    default_human_response: Incomplete
    llm: Incomplete
    vecdb: Incomplete
    parser: Incomplete
    callbacks: Incomplete
    def __init__(self, config: AgentConfig = ...) -> None: ...
    def entity_responders(
        self,
    ) -> list[
        tuple[Entity, Callable[[None | str | ChatDocument], None | ChatDocument]]
    ]: ...
    def entity_responders_async(
        self,
    ) -> list[
        tuple[
            Entity,
            Callable[
                [None | str | ChatDocument], Coroutine[Any, Any, None | ChatDocument]
            ],
        ]
    ]: ...
    @property
    def indent(self) -> str: ...
    @indent.setter
    def indent(self, value: str) -> None: ...
    def update_dialog(self, prompt: str, output: str) -> None: ...
    def get_dialog(self) -> list[tuple[str, str]]: ...
    def clear_dialog(self) -> None: ...
    def enable_message_handling(
        self, message_class: type[ToolMessage] | None = None
    ) -> None: ...
    def disable_message_handling(
        self, message_class: type[ToolMessage] | None = None
    ) -> None: ...
    def sample_multi_round_dialog(self) -> str: ...
    def create_agent_response(self, content: str | None = None) -> ChatDocument: ...
    async def agent_response_async(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def agent_response(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def create_user_response(self, content: str | None = None) -> ChatDocument: ...
    async def user_response_async(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def user_response(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def llm_can_respond(self, message: Incomplete | None = None): ...
    def create_llm_response(self, content: str | None = None) -> ChatDocument: ...
    async def llm_response_async(self, msg: Incomplete | None = None): ...
    def llm_response(self, msg: Incomplete | None = None): ...
    def has_tool_message_attempt(self, msg: str | ChatDocument | None) -> bool: ...
    def get_tool_messages(self, msg: str | ChatDocument) -> list[ToolMessage]: ...
    def get_json_tool_messages(self, input_str: str) -> list[ToolMessage]: ...
    def get_function_call_class(self, msg: ChatDocument) -> ToolMessage | None: ...
    def tool_validation_error(self, ve: ValidationError) -> str: ...
    def handle_message(self, msg: str | ChatDocument) -> None | str | ChatDocument: ...
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None: ...
    def handle_tool_message(self, tool: ToolMessage) -> None | str | ChatDocument: ...
    def num_tokens(self, prompt: str | list[LLMMessage]) -> int: ...
    def update_token_usage(
        self,
        response: LLMResponse,
        prompt: str | list[LLMMessage],
        stream: bool,
        chat: bool = True,
        print_response_stats: bool = True,
    ) -> None: ...
    def compute_token_cost(self, prompt: int, completion: int) -> float: ...
    def ask_agent(
        self,
        agent: Agent,
        request: str,
        no_answer: str = ...,
        user_confirm: bool = True,
    ) -> str | None: ...
