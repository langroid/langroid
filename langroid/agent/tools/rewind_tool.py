"""
The `rewind_tool` is used to rewind to the `n`th previous Assistant message
and replace it with a new `content`. This is useful in several scenarios and
- saves token-cost + inference time,
- reduces distracting clutter in chat history, which helps improve response quality.

This is intended to mimic how a human user might use a chat interface, where they
go down a conversation path, and want to go back in history to "edit and re-submit"
a previous message, to get a better response.

See usage examples in `tests/main/test_rewind_tool.py`.
"""

from typing import List, Tuple

import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage


def prune_messages(agent: ChatAgent, idx: int) -> ChatDocument | None:
    """
    Clear the message history of agent, starting at index `idx`,
    taking care to first clear all dependent messages (possibly from other agents'
    message histories) that are linked to the message at `idx`, via the `child_id` field
    of the `metadata` field of the ChatDocument linked from the message at `idx`.

    Args:
        agent (ChatAgent): The agent whose message history is to be pruned.
        idx (int): The index from which to start clearing the message history.

    Returns:
        The parent ChatDocument of the ChatDocument linked from the message at `idx`,
        if it exists, else None.

    """
    assert idx >= 0, "Invalid index for message history!"
    chat_doc_id = agent.message_history[idx].chat_document_id
    chat_doc = ChatDocument.from_id(chat_doc_id)
    assert chat_doc is not None, "ChatDocument not found in registry!"

    parent = ChatDocument.from_id(chat_doc.metadata.parent_id)  # may be None
    # We're invaliding the msg at idx,
    # so starting with chat_doc, go down the child links
    # and clear history of each agent, to the msg_idx
    curr_doc = chat_doc
    while child_doc := curr_doc.metadata.child:
        if child_doc.metadata.msg_idx >= 0:
            child_agent = ChatAgent.from_id(child_doc.metadata.agent_id)
            if child_agent is not None:
                child_agent.clear_history(child_doc.metadata.msg_idx)
        curr_doc = child_doc

    # Clear out ObjectRegistry entries for this ChatDocuments
    # and all descendants (in case they weren't already cleared above)
    ChatDocument.delete_id(chat_doc.id())

    # Finally, clear this agent's history back to idx,
    # and replace the msg at idx with the new content
    agent.clear_history(idx)
    return parent


class RewindTool(ToolMessage):
    """
    Used by LLM to rewind (i.e. backtrack) to the `n`th Assistant message
    and replace with a new msg.
    """

    request: str = "rewind_tool"
    purpose: str = """
        To rewind the conversation and replace the 
        <n>'th Assistant message with <content>
        """
    n: int
    content: str

    @classmethod
    def examples(cls) -> List["ToolMessage" | Tuple[str, "ToolMessage"]]:
        return [
            cls(n=1, content="What are the 3 major causes of heart disease?"),
            (
                """
                Based on the conversation so far, I realize I would get a better
                response from Bob if rephrase my 2nd message to him to: 
                'Who wrote the book Grime and Banishment?'
                """,
                cls(n=2, content="who wrote the book 'Grime and Banishment'?"),
            ),
        ]

    def response(self, agent: ChatAgent) -> str | ChatDocument:
        """
        Define the tool-handler method for this tool here itself,
        since it is a generic tool whose functionality should be the
        same for any agent.

        When LLM has correctly used this tool, rewind this agent's
        `message_history` to the `n`th assistant msg, and replace it with `content`.
        We need to mock it as if the LLM is sending this message.

        Within a multi-agent scenario, this also means that any other messages dependent
        on this message will need to be invalidated --
        so go down the chain of child messages and clear each agent's history
        back to the `msg_idx` corresponding to the child message.

        Returns:
            (ChatDocument): with content set to self.content.
        """
        idx = agent.nth_message_idx_with_role(lm.Role.ASSISTANT, self.n)
        if idx < 0:
            # set up a corrective message from AGENT
            msg = f"""
                Could not rewind to {self.n}th Assistant message!
                Please check the value of `n` and try again.
                Or it may be too early to use the `rewind_tool`.
                """
            return agent.create_agent_response(msg)

        parent = prune_messages(agent, idx)

        # create ChatDocument with new content, to be returned as result of this tool
        result_doc = agent.create_llm_response(self.content)
        result_doc.metadata.parent_id = "" if parent is None else parent.id()
        result_doc.metadata.agent_id = agent.id
        result_doc.metadata.msg_idx = idx

        # replace the message at idx with this new message
        agent.message_history.extend(ChatDocument.to_LLMMessage(result_doc))

        # set the replaced doc's parent's child to this result_doc
        if parent is not None:
            # first remove the this parent's child from registry
            ChatDocument.delete_id(parent.metadata.child_id)
            parent.metadata.child_id = result_doc.id()
        return result_doc
