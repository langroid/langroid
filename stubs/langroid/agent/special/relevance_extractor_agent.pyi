from _typeshed import Incomplete

from langroid.agent.chat_agent import (
    ChatAgent as ChatAgent,
)
from langroid.agent.chat_agent import (
    ChatAgentConfig as ChatAgentConfig,
)
from langroid.agent.chat_document import ChatDocument as ChatDocument
from langroid.agent.tools.segment_extract_tool import (
    SegmentExtractTool as SegmentExtractTool,
)
from langroid.language_models.base import LLMConfig as LLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig as OpenAIGPTConfig
from langroid.mytypes import Entity as Entity
from langroid.parsing.utils import (
    extract_numbered_segments as extract_numbered_segments,
)
from langroid.parsing.utils import (
    number_segments as number_segments,
)
from langroid.utils.constants import DONE as DONE
from langroid.utils.constants import NO_ANSWER as NO_ANSWER

console: Incomplete
logger: Incomplete

class RelevanceExtractorAgentConfig(ChatAgentConfig):
    llm: LLMConfig | None
    segment_length: int
    query: str
    system_message: str

class RelevanceExtractorAgent(ChatAgent):
    config: Incomplete
    numbered_passage: Incomplete
    def __init__(self, config: RelevanceExtractorAgentConfig) -> None: ...
    def llm_response(self, message: Incomplete | None = None): ...
    async def llm_response_async(self, message: Incomplete | None = None): ...
    def extract_segments(self, msg: SegmentExtractTool) -> str: ...
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None: ...
