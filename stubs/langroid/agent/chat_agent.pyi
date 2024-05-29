from _typeshed import Incomplete

from langroid.agent.base import (
    Agent as Agent,
)
from langroid.agent.base import (
    AgentConfig as AgentConfig,
)
from langroid.agent.base import (
    noop_fn as noop_fn,
)
from langroid.agent.chat_document import ChatDocument as ChatDocument
from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.base import (
    LLMFunctionSpec as LLMFunctionSpec,
)
from langroid.language_models.base import (
    LLMMessage as LLMMessage,
)
from langroid.language_models.base import (
    LLMResponse as LLMResponse,
)
from langroid.language_models.base import (
    Role as Role,
)
from langroid.language_models.base import (
    StreamingIfAllowed as StreamingIfAllowed,
)
from langroid.language_models.openai_gpt import OpenAIGPT as OpenAIGPT
from langroid.utils.configuration import settings as settings
from langroid.utils.output import status as status

console: Incomplete
logger: Incomplete

class ChatAgentConfig(AgentConfig):
    system_message: str
    user_message: str | None
    use_tools: bool
    use_functions_api: bool

class ChatAgent(Agent):
    config: Incomplete
    message_history: Incomplete
    tool_instructions_added: bool
    system_message: Incomplete
    user_message: Incomplete
    system_tool_instructions: str
    system_json_tool_instructions: str
    llm_functions_map: Incomplete
    llm_functions_handled: Incomplete
    llm_functions_usable: Incomplete
    llm_function_force: Incomplete
    def __init__(
        self, config: ChatAgentConfig = ..., task: list[LLMMessage] | None = None
    ) -> None: ...
    def clone(self, i: int = 0) -> ChatAgent: ...
    def set_system_message(self, msg: str) -> None: ...
    def set_user_message(self, msg: str) -> None: ...
    @property
    def task_messages(self) -> list[LLMMessage]: ...
    def clear_history(self, start: int = -2) -> None: ...
    def update_history(self, message: str, response: str) -> None: ...
    def json_format_rules(self) -> str: ...
    def tool_instructions(self) -> str: ...
    def augment_system_message(self, message: str) -> None: ...
    def last_message_with_role(self, role: Role) -> LLMMessage | None: ...
    def update_last_message(self, message: str, role: str = ...) -> None: ...
    def enable_message(
        self,
        message_class: type[ToolMessage] | None,
        use: bool = True,
        handle: bool = True,
        force: bool = False,
        require_recipient: bool = False,
        include_defaults: bool = True,
    ) -> None: ...
    def disable_message_handling(
        self, message_class: type[ToolMessage] | None = None
    ) -> None: ...
    def disable_message_use(self, message_class: type[ToolMessage] | None) -> None: ...
    def disable_message_use_except(self, message_class: type[ToolMessage]) -> None: ...
    def llm_response(
        self, message: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    async def llm_response_async(
        self, message: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def llm_response_messages(
        self, messages: list[LLMMessage], output_len: int | None = None
    ) -> ChatDocument: ...
    async def llm_response_messages_async(
        self, messages: list[LLMMessage], output_len: int | None = None
    ) -> ChatDocument: ...
    def llm_response_forget(self, message: str) -> ChatDocument: ...
    async def llm_response_forget_async(self, message: str) -> ChatDocument: ...
    def chat_num_tokens(self, messages: list[LLMMessage] | None = None) -> int: ...
    def message_history_str(self, i: int | None = None) -> str: ...
