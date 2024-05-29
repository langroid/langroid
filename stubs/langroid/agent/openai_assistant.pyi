from enum import Enum
from typing import Any

from _typeshed import Incomplete
from openai.types.beta import Assistant as Assistant
from openai.types.beta import Thread as Thread
from openai.types.beta.threads import Message
from openai.types.beta.threads import Run as Run
from openai.types.beta.threads.runs import RunStep as RunStep
from pydantic import BaseModel

from langroid.agent.chat_agent import (
    ChatAgent as ChatAgent,
)
from langroid.agent.chat_agent import (
    ChatAgentConfig as ChatAgentConfig,
)
from langroid.agent.chat_document import ChatDocument as ChatDocument
from langroid.agent.tool_message import ToolMessage as ToolMessage
from langroid.language_models.base import (
    LLMFunctionCall as LLMFunctionCall,
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
from langroid.language_models.openai_gpt import (
    OpenAIChatModel as OpenAIChatModel,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPT as OpenAIGPT,
)
from langroid.language_models.openai_gpt import (
    OpenAIGPTConfig as OpenAIGPTConfig,
)
from langroid.utils.configuration import settings as settings
from langroid.utils.system import (
    generate_user_id as generate_user_id,
)
from langroid.utils.system import (
    update_hash as update_hash,
)

logger: Incomplete

class ToolType(str, Enum):
    RETRIEVAL: str
    CODE_INTERPRETER: str
    FUNCTION: str

class AssistantTool(BaseModel):
    type: ToolType
    function: dict[str, Any] | None
    def dct(self) -> dict[str, Any]: ...

class AssistantToolCall(BaseModel):
    id: str
    type: ToolType
    function: LLMFunctionCall

class RunStatus(str, Enum):
    QUEUED: str
    IN_PROGRESS: str
    COMPLETED: str
    REQUIRES_ACTION: str
    EXPIRED: str
    CANCELLING: str
    CANCELLED: str
    FAILED: str
    TIMEOUT: str

class OpenAIAssistantConfig(ChatAgentConfig):
    use_cached_assistant: bool
    assistant_id: str | None
    use_tools: bool
    use_functions_api: bool
    use_cached_thread: bool
    thread_id: str | None
    cache_responses: bool
    timeout: int
    llm: Incomplete
    tools: list[AssistantTool]
    files: list[str]

class OpenAIAssistant(ChatAgent):
    config: Incomplete
    llm: Incomplete
    client: Incomplete
    runs: Incomplete
    threads: Incomplete
    thread_messages: Incomplete
    assistants: Incomplete
    pending_tool_ids: Incomplete
    cached_tool_ids: Incomplete
    thread: Incomplete
    assistant: Incomplete
    run: Incomplete
    def __init__(self, config: OpenAIAssistantConfig) -> None: ...
    files: Incomplete
    def add_assistant_files(self, files: list[str]) -> None: ...
    def add_assistant_tools(self, tools: list[AssistantTool]) -> None: ...
    def enable_message(
        self,
        message_class: type[ToolMessage] | None,
        use: bool = True,
        handle: bool = True,
        force: bool = False,
        require_recipient: bool = False,
        include_defaults: bool = True,
    ) -> None: ...
    @staticmethod
    def thread_msg_to_llm_msg(msg: Message) -> LLMMessage: ...
    def set_system_message(self, msg: str) -> None: ...
    def process_citations(self, thread_msg: Message) -> None: ...
    def llm_response(
        self, message: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    async def llm_response_async(
        self, message: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
    def agent_response(
        self, msg: str | ChatDocument | None = None
    ) -> ChatDocument | None: ...
