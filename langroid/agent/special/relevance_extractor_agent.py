"""
Agent to retrieve relevant segments from a body of text,
that are relevant to a query.

"""

import logging
from typing import Optional, no_type_check

from rich.console import Console

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tools.segment_extract_tool import SegmentExtractTool
from langroid.language_models.base import LLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.utils import extract_numbered_segments, number_segments
from langroid.utils.constants import DONE, NO_ANSWER

console = Console()
logger = logging.getLogger(__name__)


class RelevanceExtractorAgentConfig(ChatAgentConfig):
    llm: LLMConfig | None = OpenAIGPTConfig()
    segment_length: int = 1  # number of sentences per segment
    query: str = ""  # query for relevance extraction
    system_message = """
    The user will give you a PASSAGE containing segments numbered as  
    <#1#>, <#2#>, <#3#>, etc.,
    followed by a QUERY. Extract ONLY the segment-numbers from 
    the PASSAGE that are RELEVANT to the QUERY.
    Present the extracted segment-numbers using the `extract_segments` tool/function.
    """


class RelevanceExtractorAgent(ChatAgent):
    """
    Agent for extracting segments from text, that are relevant to a given query.
    """

    def __init__(self, config: RelevanceExtractorAgentConfig):
        super().__init__(config)
        self.config: RelevanceExtractorAgentConfig = config
        self.enable_message(SegmentExtractTool)
        self.numbered_passage: Optional[str] = None

    @no_type_check
    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """Compose a prompt asking to extract relevant segments from a passage.
        Steps:
        - number the segments in the passage
        - compose prompt
        - send to LLM
        """
        assert self.config.query is not None, "No query specified"
        assert message is not None, "No message specified"
        message_str = message.content if isinstance(message, ChatDocument) else message
        # number the segments in the passage
        self.numbered_passage = number_segments(message_str, self.config.segment_length)
        # compose prompt
        prompt = f"""
        PASSAGE:
        {self.numbered_passage}
        
        QUERY: {self.config.query}
        """
        # send to LLM
        return super().llm_response(prompt)

    @no_type_check
    async def llm_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """
        Compose a prompt asking to extract relevant segments from a passage.
        Steps:
        - number the segments in the passage
        - compose prompt
        - send to LLM
        The LLM is expected to generate a structured msg according to the
        SegmentExtractTool schema, i.e. it should contain a `segment_list` field
        whose value is a list of segment numbers or ranges, like "10,12,14-17".
        """

        assert self.config.query is not None, "No query specified"
        assert message is not None, "No message specified"
        message_str = message.content if isinstance(message, ChatDocument) else message
        # number the segments in the passage
        self.numbered_passage = number_segments(message_str, self.config.segment_length)
        # compose prompt
        prompt = f"""
        PASSAGE:
        {self.numbered_passage}
        
        QUERY: {self.config.query}
        """
        # send to LLM
        return await super().llm_response_async(prompt)

    def extract_segments(self, msg: SegmentExtractTool) -> str:
        """Method to handle a segmentExtractTool message from LLM"""
        spec = msg.segment_list
        if len(self.message_history) == 0:
            return DONE + " " + NO_ANSWER
        if spec is None or spec.strip() in ["", NO_ANSWER]:
            return DONE + " " + NO_ANSWER
        assert self.numbered_passage is not None, "No numbered passage"
        # assume this has numbered segments
        try:
            extracts = extract_numbered_segments(self.numbered_passage, spec)
        except Exception:
            return DONE + " " + NO_ANSWER
        # this response ends the task by saying DONE
        return DONE + " " + extracts

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Handle case where LLM forgets to use SegmentExtractTool"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            return DONE + " " + NO_ANSWER
        else:
            return None
