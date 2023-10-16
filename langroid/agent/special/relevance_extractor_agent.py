"""
Agent to retrieve relevant sentences from a body of text,
that are relevant to a query.

"""
import logging

from rich.console import Console

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.sentence_extract_tool import SentenceExtractTool
from langroid.language_models.base import Role
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.parsing.utils import extract_numbered_sentences

console = Console()
logger = logging.getLogger(__name__)


class RelevanceExtractorAgentConfig(ChatAgentConfig):
    llm: OpenAIGPTConfig = OpenAIGPTConfig()
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

    def extract_sentences(self, msg: SentenceExtractTool) -> str:
        spec = msg.sentence_list
        if len(self.message_history) == 0:
            return ""
        if spec is None:
            return ""
        assert len(self.message_history) > 0, "No message history"
        assert self.message_history[-1].role == Role.ASSISTANT
        last_user_msg = self.last_message_with_role(Role.USER)
        if last_user_msg is None:
            return ""
        # assume this has numbered sentences
        extracts = extract_numbered_sentences(last_user_msg.content, spec)
        # this response ends the task by saying DONE
        return "DONE " + extracts
