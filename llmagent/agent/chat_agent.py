from llmagent.language_models.base import LLMMessage, Role, StreamingIfAllowed
from llmagent.agent.base import Agent, AgentConfig, AgentMessage
from llmagent.mytypes import Document
from llmagent.utils.configuration import settings
from typing import List, Optional, Type
from rich import print


class ChatAgent(Agent):
    """
    Chat Agent interacting with external env
    (could be human, or external tools).
    The agent (the LLM actually) is provided with a "Task Spec", and told to think in
    small steps. It may be given a set of possible "Actions", and if so it is told to
    emit the appropriate action in each round. Each round consists of:
    - LLM emits an Action, or says Done
    - LLM receives an Observation from that Action
        (which could be a human response but not necessarily)
    - LLM thinks
    """

    def __init__(self, config: AgentConfig, task: Optional[List[LLMMessage]] = None):
        """
        Chat-mode agent initialized with task spec as the initial message sequence
        Args:
            config: settings for the agent
            task: seq of messages to start with. If empty a "system" msg is
                constructed by default.
                Note these messages are not yet issued to LLM at agent init.

        !!! note
             `self.message_history` is different from `self.dialog` (in Agent class):

            - `self.message_history` is the sequence of messages sent to the LLM in
            **chat mode** (e.g. when using OpenAI `ChatCompletion.create()`)
                Typically we send a sequence of such messages to "prime"
            the LLM context for some task, and we extend and re-send this sequence to
            continue interaction. Note that consecutive messages in the sequence could
            have different or same roles (e.g. "user", "assistant"). Each message has a
            "dict" structure, which we call :class:`LLMMessage`.

            - `self.dialog` is the sequence of `(prompt, response)` tuples produced
            when interacting with an LLM in **completion mode**,
            where `prompt (str)` is sent TO the LLM, and `response (str)` is received
            FROM the LLM. Typically as an LLM conversation goes on, we collate
            `self.dialog` into a single string, and insert it into the context part
            of the next prompt to the LLM.

        """
        super().__init__(config)
        self.message_history: List[LLMMessage] = []
        self.json_instructions_idx: int = -1
        if task is None:
            task = [LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant")]
        self.task_messages = task

    def enable_message(self, message_class: Type[AgentMessage]) -> None:
        super().enable_message(message_class)
        self.update_message_instructions()

    def disable_message(self, message_class: Type[AgentMessage]) -> None:
        super().disable_message(message_class)
        self.update_message_instructions()

    def update_message_instructions(self) -> None:
        """
        Add special instructions on situations when the LLM should send JSON-formatted
        messages, and save the index position of these instructions in the
        message history.
        """
        # note according to the openai-cookbook, GPT-3.5 pays less attention to the
        # system messages, so we add the instructions as a user message
        # TODO need to adapt this based on model type.
        json_instructions = super().message_format_instructions()
        if self.json_instructions_idx < 0:
            self.task_messages.append(
                LLMMessage(role=Role.USER, content=json_instructions)
            )
            self.json_instructions_idx = len(self.task_messages) - 1
        else:
            self.task_messages[self.json_instructions_idx].content = json_instructions

        # Note that task_messages is the initial set of messages created to set up
        # the task, and they may not yet have been sent to the LLM at this point.

        # But if the task_messages have already been sent to the LLM, then we need to
        # update the self.message_history as well, since this history will be sent to
        # the LLM on each round, after appending the latest assistant, user msgs.
        if len(self.message_history) > 0:
            self.message_history[self.json_instructions_idx].content = json_instructions

    def run(
        self, iters: int = -1, default_human_response: Optional[str] = None
    ) -> None:
        """
        Run the agent in chat mode, until the user types "exit", "quit", "q", "x",
        "bye", or, when iters > 0,  until this number of iterations is reached.
        Args:
            iters: number of iterations to run the agent for. If -1, run until user
                types "exit", "quit", "q", "x", "bye"
            default_human_response: if not None, this means we are running this
                agent without human input, and we use this string as the default
                human response when the agent's `handle_method` returns None.
                This can be useful for automated/non-interactive testing.
        """
        llm_msg = self.start().content
        if settings.debug:
            print(f"[red]{self.message_history_str()}")
        niters = 0
        while True:
            niters += 1
            if iters > 0 and niters > iters:
                break
            agent_result = self.handle_message(llm_msg)
            if agent_result is not None:
                msg = f"{agent_result}"
                print(f"[red]Agent: {agent_result}")
            else:
                if default_human_response is not None:
                    msg = default_human_response
                    print(f"[blue]Human: {default_human_response}")
                else:
                    print("\n[blue]Human: ", end="")
                    msg = input("")
            if msg in ["exit", "quit", "q", "x", "bye"]:
                print("[green] Bye, hope this was useful!")
                break
            llm_msg = self.respond(msg).content

    def start(self) -> Document:
        """
        Start the agent, by sending the initial task spec to LLM
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        with StreamingIfAllowed(self.llm):
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
        with StreamingIfAllowed(self.llm):
            response = self.respond_messages(self.message_history)
        self.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=response.content)
        )
        return Document(content=response.content, metadata=response.metadata)

    def message_history_str(self, i: Optional[int] = None) -> str:
        """
        Return a string representation of the message history
        Args:
            i: if provided, return only the i-th message when i is postive,
                or last k messages when i = -k.
        Returns:
        """
        if i is None:
            return "\n".join([str(m) for m in self.message_history])
        elif i > 0:
            return str(self.message_history[i])
        else:
            return "\n".join([str(m) for m in self.message_history[i:]])
