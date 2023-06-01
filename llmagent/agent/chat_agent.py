from llmagent.language_models.base import LLMMessage, Role, StreamingIfAllowed
from llmagent.agent.base import Agent, AgentConfig, AgentMessage
from llmagent.mytypes import Document, DocMetaData
from llmagent.utils.configuration import settings
from typing import List, Optional, Type
from llmagent.agent.base import Entity
from rich import print
from contextlib import ExitStack
import logging
from rich.console import Console

console = Console()

logger = logging.getLogger(__name__)


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

    def clear_history(self, start: int = -2) -> None:
        """
        Clear the message history, starting at the index `start`

        Args:
            start (int): index of first message to delete; default = -2
                    (i.e. delete last 2 messages, typically these
                    are the last user and assistant messages)
        """
        n = len(self.message_history)
        if start < 0:
            start = max(0, n + start)
        self.message_history = self.message_history[:start]

    def update_history(self, message: str, response: str) -> None:
        """
        Update the message history with the latest user message and LLM response.
        Args:
            message (str): user message
            response: (str): LLM response
        """
        self.message_history.extend(
            [
                LLMMessage(role=Role.USER, content=message),
                LLMMessage(role=Role.ASSISTANT, content=response),
            ]
        )

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the message history.
        Args:
            message (str): user message
        """
        if len(self.message_history) > 0:
            self.message_history.append(LLMMessage(role=Role.USER, content=message))
        else:
            self.task_messages.append(LLMMessage(role=Role.USER, content=message))

    def update_last_message(self, message: str, role: str = Role.USER) -> None:
        """
        Update the last message with role `role` in the message history.
        Useful when we want to replace a long user prompt, that may contain context
        documents plus a question, with just the question.
        Args:
            message (str): user message
            role (str): role of message to replace
        """
        if len(self.message_history) == 0:
            return
        # find last message in self.message_history with role `role`
        for i in range(len(self.message_history) - 1, -1, -1):
            if self.message_history[i].role == role:
                self.message_history[i].content = message
                break

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

    def setup_task(
        self, msg: str = None, ent: Entity = Entity.USER, system_message: str = None
    ):
        """
        Set up the task by sending the initial messages to the LLM.

        Args:
            msg (str): user message (including instructions, initial question etc)
            ent (Entity): entity to use for the first message
            system_message (str): system message containing role etc.
        """
        if system_message is not None:
            self.task_messages[0].content = system_message
        if msg is None:
            assert (
                len(self.task_messages) > 1 and self.task_messages[1].role == Role.USER
            ), """
                message can be None only if there is at least one 
                USER message in self.task_messages. 
                """
            msg = self.task_messages[-1].content
        super().setup_task(msg, ent=ent)

    def do_task(
        self,
        msg: str = None,
        system_message: str = None,
        rounds: int = None,
    ) -> Optional[Document]:
        """
        Do the task, as specified in the optional msg
        (if absent use the self.task_messages),

        Args:
            msg (str): optional initial msg from user
            system_message: optional system message spe
            rounds: how many rounds to run the task for

        Returns:
            Document: result in the form of a Document object
        """
        self.setup_task(msg, ent=Entity.USER, system_message=system_message)
        return super().do_task(msg, rounds=rounds)

    def llm_response(self, message: str = None) -> Document:
        """
        Respond to a single user message, appended to the message history,
        in "chat" mode
        Args:
            message: user message; if None, use the self.task_messages
        Returns:
        """
        assert (
            message is not None or len(self.message_history) == 0
        ), "message can be None only if message_history is empty, i.e. at start."

        if len(self.message_history) == 0:
            # task_messages have not yet been loaded, so load them
            self.message_history = self.task_messages
            # for debugging, show the initial message history
            if settings.debug:
                print(
                    f"""
                [red]LLM Initial Msg History:
                {self.message_history_str()}
                """
                )

        if message is not None:
            self.message_history.append(LLMMessage(role=Role.USER, content=message))

        hist = self.message_history
        output_len = self.config.llm.max_output_tokens
        if (
            self.chat_num_tokens(hist)
            > self.llm.chat_context_length() - self.config.llm.max_output_tokens
        ):
            # chat + output > max context length,
            # so first try to shorten requested output len to fit.
            output_len = self.llm.chat_context_length() - self.chat_num_tokens(hist)
            if output_len < self.config.llm.min_output_tokens:
                # unacceptably small output len, so drop early parts of conv history
                # if output_len is still too long, then drop early parts of conv history
                # TODO we should really be doing summarization or other types of
                #   prompt-size reduction
                while (
                    self.chat_num_tokens(hist)
                    > self.llm.chat_context_length() - self.config.llm.min_output_tokens
                ):
                    # try dropping early parts of conv history
                    # TODO we should really be doing summarization or other types of
                    #   prompt-size reduction
                    if len(hist) <= 2:
                        # first two are "reserved" for the initial system, user msgs
                        # that presumably set up the initial "priming" or "task" for
                        # the agent.
                        raise ValueError(
                            """
                        The message history is longer than the max chat context 
                        length allowed, and we have run out of messages to drop."""
                        )
                    hist = hist[:2] + hist[3:]

                if len(hist) < len(self.message_history):
                    msg_tokens = self.chat_num_tokens()
                    logger.warning(
                        f"""
                    Chat Model context length is {self.llm.chat_context_length()} 
                    tokens, but the current message history is {msg_tokens} tokens long.
                    Dropped the {len(self.message_history) - len(hist)} messages
                    from early in the conversation history so total tokens are 
                    low enough to allow minimum output length of 
                    {self.config.llm.min_output_tokens} tokens.
                    """
                    )

        with StreamingIfAllowed(self.llm):
            response = self.llm_response_messages(hist, output_len)
        self.message_history.append(
            LLMMessage(role=Role.ASSISTANT, content=response.content)
        )
        return Document(content=response.content, metadata=response.metadata)

    def llm_response_messages(
        self, messages: List[LLMMessage], output_len: int = None
    ) -> Document:
        """
        Respond to a series of messages, e.g. with OpenAI ChatCompletion
        Args:
            messages: seq of messages (with role, content fields) sent to LLM
        Returns:
            Document (i.e. with fields "content", "metadata")
        """
        output_len = output_len or self.config.llm.max_output_tokens
        with ExitStack() as stack:  # for conditionally using rich spinner
            if not self.llm.get_stream():
                # show rich spinner only if not streaming!
                cm = console.status("LLM responding to messages...")
                stack.enter_context(cm)
            response = self.llm.chat(messages, output_len)
        displayed = False
        if not self.llm.get_stream() or response.cached:
            displayed = True
            cached = "[red](cached)[/red]" if response.cached else ""
            print(cached + "[green]" + response.message)
        return Document(
            content=response.message,
            metadata=DocMetaData(
                source=Entity.LLM.value,
                sender=Entity.LLM.value,
                usage=response.usage,
                displayed=displayed,
                cached=response.cached,
            ),
        )

    def _llm_response_temp_context(self, message: str, prompt: str) -> Document:
        """
        Get LLM response to `prompt` (which presumably includes the `message`
        somewhere, along with possible large "context" passages),
        but only include `message` as the USER message, and not the
        full `prompt`, in the message history.
        Args:
            message: the original, relatively short, user request or query
            prompt: the full prompt potentially containing `message` plus context

        Returns:
            Document object containing the response.
        """
        # we explicitly call THIS class's respond method,
        # not a derived class's (or else there would be infinite recursion!)
        answer_doc = ChatAgent.llm_response(self, prompt)
        self.update_last_message(message, role=Role.USER)
        return answer_doc

    def llm_response_forget(self, message: str) -> Document:
        """
        LLM Response to single message, and restore message_history.
        In effect a "one-off" message & response that leaves agent
        message history state intact.

        Args:
            message (str): user message

        Returns:
            A Document object with the response.

        """
        # explicitly call THIS class's respond method,
        # not a derived class's (or else there would be infinite recursion!)
        response = ChatAgent.llm_response(self, message)
        # clear the last two messages, which are the
        # user message and the assistant response
        self.message_history.pop()
        self.message_history.pop()
        return response

    def chat_num_tokens(self, messages: Optional[List[LLMMessage]] = None) -> int:
        """
        Total number of tokens in the message history so far.

        Args:
            messages: if provided, compute the number of tokens in this list of
                messages, rather than the current message history.
        Returns:
            int: number of tokens in message history
        """
        hist = messages if messages is not None else self.message_history
        return sum([self.parser.num_tokens(m.content) for m in hist])

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

    def reformat_message(self, msg: str) -> str:
        """
        Reformat a message according to any applicable tool (message class).
        If no tool is applicable, return the original message, or an equivalent.
        Args:
            msg: original message to be reformatted

        Returns:
            Reformatted message that makes use of a tool (message class) when possible.
        """
        formatter_agent = ChatAgent(self.config, task=self.task_messages)
        reformatted = formatter_agent.llm_response(msg).content
        return reformatted
