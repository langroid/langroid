from llmagent.language_models.base import LLMMessage, LLMResponse, Role
from llmagent.agent.base import Agent, AgentConfig
from llmagent.mytypes import Document
from typing import List


class COTAgent(Agent):
    """
    Chain-of-thought Agent interacting with external env
    (could be human, or external tools).
    The agent (the LLM actually) is provided with a "Task Spec", and told to think in
    small steps. It may be given a set of possible "Actions", and if so it is told to
    emit the appropriate action in each round. Each round consists of:
    - LLM emits an Action, or says Done
    - LLM receives an Observation from that Action
        (which could be a human response but not necessarily)
    - LLM thinks
    """

    def __init__(self, config: AgentConfig, task: List[LLMMessage]):
        """
        Agent initialized with task spec as the initial message sequence
        Args:
            config: settings for the agent
            task: seq of messages to start with
                Note these messages are not yet issued to LLM at agent init.
        """
        super().__init__(config)
        self.message_history: List[LLMMessage] = []
        self.task_messages = task

    def start(self) -> Document:
        """
        Start the agent, by sending the initial task spec to LLM
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        response = self.respond_messages(self.task_messages)
        self.message_history = self.task_messages + [
            LLMMessage(
                role=Role.ASSISTANT,
                content=response.content,
            )
        ]
        return Document(content=response.content, metadata=response.metadata)

    def respond(self, message: str) -> Document:
        """
        Respond to a single user message, appended to the message history,
        in "chat" mode
        Args:
            message: user message
        Returns:
        """
        self.message_history.append(LLMMessage(role=Role.USER, content=message))
        response = self.respond_messages(self.message_history)
        self.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=response.content)
        )
        return Document(content=response.content, metadata=response.metadata)
