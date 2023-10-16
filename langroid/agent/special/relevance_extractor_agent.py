"""
Agent to retrieve relevant sentences from a body of text,
that are relevant to a query.

"""
import logging
from typing import Optional, no_type_check

from rich.console import Console

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tools.sentence_extract_tool import SentenceExtractTool
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.utils import extract_numbered_sentences, number_sentences

console = Console()
logger = logging.getLogger(__name__)


class RelevanceExtractorAgentConfig(ChatAgentConfig):
    llm: OpenAIGPTConfig = OpenAIGPTConfig()
    query: str  # query for relevance extraction
    system_message = """
    The user will give you a PASSAGE containing numbered sentences, 
    followed by a QUERY. Your task is to extract the sentence-numbers from the PASSAGE
    that are relevant to the QUERY. You must use the `extract_sentences` 
    tool/function to present your answer, by setting the `sentence_list` field 
    to a list of sentence numbers or ranges, like "10,12,14-17".
    """


class RelevanceExtractorAgent(ChatAgent):
    """
    Agent for extracting sentences from text, that are relevant to a given query.
    """

    def __init__(self, config: RelevanceExtractorAgentConfig):
        super().__init__(config)
        self.config: RelevanceExtractorAgentConfig = config
        self.enable_message(SentenceExtractTool)
        self.numbered_passage: Optional[str] = None

    @no_type_check
    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        """Compose a prompt asking to extract relevant sentences from a passage.
        Steps:
        - number the sentences in the passage
        - compose prompt
        - send to LLM
        """
        assert self.config.query is not None, "No query specified"
        assert message is not None, "No message specified"
        message_str = message.content if isinstance(message, ChatDocument) else message
        # number the sentences in the passage
        self.numbered_passage = number_sentences(message_str)
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
        """Compose a prompt asking to extract relevant sentences from a passage.
        Steps:
        - number the sentences in the passage
        - compose prompt
        - send to LLM
        """
        assert self.config.query is not None, "No query specified"
        assert message is not None, "No message specified"
        message_str = message.content if isinstance(message, ChatDocument) else message
        # number the sentences in the passage
        self.numbered_passage = number_sentences(message_str)
        # compose prompt
        prompt = f"""
        PASSAGE:
        {self.numbered_passage}
        
        QUERY: {self.config.query}
        """
        # send to LLM
        return await super().llm_response_async(prompt)

    def extract_sentences(self, msg: SentenceExtractTool) -> str:
        """Method to handle a SentenceExtractTool message from LLM"""
        spec = msg.sentence_list
        if len(self.message_history) == 0:
            return ""
        if spec is None or spec.strip() == "":
            return ""
        assert self.numbered_passage is not None, "No numbered passage"
        # assume this has numbered sentences
        extracts = extract_numbered_sentences(self.numbered_passage, spec)
        # this response ends the task by saying DONE
        return "DONE " + extracts
