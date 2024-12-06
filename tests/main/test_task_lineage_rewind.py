"""
Test various "lineage" book-keeping in (multi) agent ChatDocument chains,
in metadata fields:
- parent
- child
- agent
- msg_idx
"""

from typing import Optional

import pytest

import langroid as lr
from langroid import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tools.orchestration import DoneTool
from langroid.agent.tools.rewind_tool import RewindTool, prune_messages
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.configuration import (
    Settings,
    set_global,
)
from langroid.utils.constants import DONE


class MockAgent(ChatAgent):
    def user_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        Mock user_response method for testing
        """
        txt = msg if isinstance(msg, str) else msg.content
        map = dict([("2", "3"), ("3", "5")])
        response = map.get(txt)
        # return the increment of input number
        return self.create_user_response(response)


def test_lineage_1_task():
    agent = MockAgent(
        ChatAgentConfig(
            name="Mock",
            llm=MockLMConfig(
                response_dict={
                    "1": "2",
                    "3": DoneTool(content="100").to_json(),
                },
            ),
        )
    )
    task = lr.Task(agent, interactive=True, only_user_quits_root=False)
    result = task.run("1")
    assert "100" in result.content

    # Msg history is:
    # - sys msg: helpful asst
    # - u1: user: 1
    # - a1: assistant: 2
    # - u2: user: 3
    # - a2: assistant: DoneTool(100)
    # - ag: 100
    # Then user says "q" -> this results in a pending_message update
    # Finally the task result is another ChatDocument

    assert len(agent.message_history) == 5
    msg_u1 = agent.message_history[1]
    msg_a1 = agent.message_history[2]
    msg_u2 = agent.message_history[3]
    msg_a2 = agent.message_history[4]

    # ChatDocument objects linked by each msg
    cd_u1 = ChatDocument.from_id(msg_u1.chat_document_id)
    cd_a1 = ChatDocument.from_id(msg_a1.chat_document_id)
    cd_u2 = ChatDocument.from_id(msg_u2.chat_document_id)
    cd_a2 = ChatDocument.from_id(msg_a2.chat_document_id)
    cd_ag = cd_a2.child

    assert cd_u1.parent is None
    assert cd_u1.child is cd_a1
    assert cd_u1.metadata.agent_id == agent.id
    assert cd_u1.metadata.msg_idx == 1

    assert cd_a1.parent is cd_u1
    assert cd_a1.child is cd_u2
    assert cd_a1.metadata.agent_id == agent.id
    assert cd_a1.metadata.msg_idx == 2

    assert cd_u2.parent is cd_a1
    assert cd_u2.child is cd_a2
    assert cd_u2.metadata.agent_id == agent.id
    assert cd_u2.metadata.msg_idx == 3

    assert cd_a2.parent is cd_u2
    assert cd_a2.child is cd_ag
    assert cd_ag.parent is cd_a2
    assert cd_ag is result
    assert cd_a2.metadata.agent_id == agent.id
    assert cd_a2.metadata.msg_idx == 4

    # prune messages starting at a1
    parent = prune_messages(agent, 2)
    assert parent is cd_u1
    assert len(agent.message_history) == 2

    # check that the obj registry no longer has the deleted ChatDocuments
    assert ChatDocument.from_id(cd_a1.id()) is None
    assert ChatDocument.from_id(cd_u2.id()) is None
    assert ChatDocument.from_id(cd_a2.id()) is None
    assert ChatDocument.from_id(result.id()) is None


@pytest.mark.parametrize("use_done_tool", [True, False])
def test_lineage_2_task(use_done_tool: bool):
    def done_num(num: int) -> str:
        return (
            DoneTool(content=str(num)).to_json() if use_done_tool else f"{DONE} {num}"
        )

    # set up two agents with no user interaction, only LLM talk to each other
    alice = MockAgent(
        ChatAgentConfig(
            name="Alice",
            llm=MockLMConfig(
                response_dict={
                    "1": "2",
                    "3": "4",
                    "5": "6",
                    "7": done_num(100),
                },
            ),
        )
    )

    alice_task = lr.Task(alice, interactive=False, restart=False)

    bob = MockAgent(
        ChatAgentConfig(
            name="Bob",
            llm=MockLMConfig(
                response_dict={
                    "2": done_num(3),
                    "4": done_num(5),
                    "6": done_num(7),
                    "20": done_num(30),
                    "40": done_num(50),
                    "60": done_num(70),
                },
            ),
        )
    )
    # Note we set restart=False to prevent Bob task from resetting agent history,
    # which would lose lineage.
    bob_task = lr.Task(bob, interactive=False)

    alice_task.add_sub_task(bob_task)
    result = alice_task.run("1")
    assert "100" in result.content

    # msg seq
    # - sys1: alice sys msg  A0  (Bob also has sys msg B0)
    # - au1: alice user 1    A1
    # - a1: alice 2          A2
    # - bu1: bob user 2      B1 (alice 2 comes in as User 2 to Bob task)
    # - b1: bob DONE 3       B2
    # - au2: user 3          A3 (result from Bob returned to Alice task as User)
    # - a2: alice 4          A4
    # - bu2: user 4          B3
    # - b2: bob DONE 5       B4
    # - au3: user 5          A5
    # - a3: alice 6          A6
    # - bu3: user 6          B5
    # - b3: bob DONE 7       B6
    # - au4: user 7          A7
    # - a4: alice DONE 100   A8

    alice_chat_docs = [
        ChatDocument.from_id(msg.chat_document_id)
        for msg in alice.message_history[1:]  # exclude sys msg
    ]
    bob_chat_docs = [
        ChatDocument.from_id(msg.chat_document_id)
        for msg in bob.message_history[1:]  # exclude sys msg
    ]
    # prune Alice msgs starting at A2
    parent = prune_messages(alice, 2)

    assert len(alice.message_history) == 2
    assert parent is alice_chat_docs[0]  # sys msg has no chat doc

    # all of Alice's chat docs starting from A2 (idx 1) should be absent in registry
    assert all(ChatDocument.from_id(cd.id()) is None for cd in alice_chat_docs[1:])
    # none of Bob's chat docs should be in registry
    assert all(ChatDocument.from_id(cd.id()) is None for cd in bob_chat_docs)

    # continue running the alice task, with a new response dict;
    # instead of 1 -> 2, do 1 -> 20, and continue in a similar path
    # but all numbers are 10x the previous ones.
    # This leads to a different conversation path, of the same length,
    # but with final result of 200 instead of 100
    alice.llm.config.response_dict = {
        "1": "20",
        "30": "40",
        "50": "60",
        "70": done_num(200),
    }
    result = alice_task.run()
    assert "200" in result.content

    assert len(alice.message_history) == 9
    assert len(bob.message_history) == 7

    alice.llm.config.response_dict = {
        "1": "2",
        "3": "4",
        "5": "6",
        "7": done_num(100),
    }

    # manually apply rewind tool:
    # rewind alice to 1st asst msg, to say "2" instead of "20"
    rewind_tool = RewindTool(n=1, content="2")
    new_llm_response = rewind_tool.response(alice)
    assert new_llm_response.content == "2"
    assert new_llm_response.metadata.sender == lr.Entity.LLM
    # continue running alice task with this new response...
    result = alice_task.run()
    # ... it should end with final result of 100
    assert "100" in result.content
    assert len(alice.message_history) == 9
    assert len(bob.message_history) == 7

    # Have the LLM use the rewind tool directly, to change the response
    alice.llm.config.response_dict = {
        "1": "2",
        "3": "4",
        "5": "6",
        "7": "{'request': 'rewind_tool', 'n': 1, 'content': '20'}",
        "30": "40",
        "50": "60",
        "70": done_num(200),
    }

    alice.enable_message(RewindTool)
    alice.clear_history(0)
    bob.clear_history(0)
    result = alice_task.run("1")
    assert "200" in result.content


@pytest.mark.parametrize("use_done_tool", [True, False])
def test_rewind_tool(test_settings: Settings, use_done_tool: bool):
    set_global(test_settings)

    done_tool_name = DoneTool.default_value("request")
    if use_done_tool:
        done_response = f"""
        use the TOOL: `{done_tool_name}` with `content` field set to your final answer
        """
    else:
        done_response = f"say {DONE} and show your final answer"

    alice = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Alice",
            system_message=f"""
            You only know english and have absolutely no calculation abilities.
            
            You are trying to find out the Polinsky transform of the number 5.
            You don't know how to compute it so you simply ask your helper Bob,
            saying "What is the Polinsky transform of 5?".
            You may not get an answer right away, so ask in different ways,
            e.g. "Please compute Polinsky(5)", etc.
            
            If you don't get a good answer after 3 tries, do this:
            
            You realize that the Polinksy transform of x is simply 3x+1,
            so use the `rewind_tool` to replace your 1st message with a 
            new way of asking.
            
            Remember you MUST use the `rewind_tool` when you ask in this new way,
            and ONLY after 3 unsuccessful tries with the 
            original question and its variants.

            Once you are satisfied with the answers, 
            {done_response}
            """,
        )
    )

    bob = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Bob",
            system_message="""
            The user will ask you about some computation. 
            BE HONEST: if you know how to calculate it, show the answer,
            OTHERWISE simply say "Cannot Do".
            """,
        )
    )

    alice_task = lr.Task(alice, interactive=False)
    bob_task = lr.Task(bob, interactive=False, single_round=True)
    alice_task.add_sub_task(bob_task)
    alice.enable_message(RewindTool)

    # With weaker LLM (even GPT-4o sometimes), Alice may continue
    # to use RewindTool even after Bob has given the answer,
    # so we limit the number of turns to 12 ...
    alice_task.run(turns=12)
    assert any("16" in m.content for m in bob.message_history)

    # ... and truncate Bob's message history to the point where
    # he responds with "16" to Alice's question.

    # Find index of earliest Bob msg that has "16" in it
    bob_msg_idx = next(
        i for i, m in enumerate(bob.message_history) if "16" in m.content
    )
    bob_hist = bob.message_history[: bob_msg_idx + 1]
    # If rewind used correctly, new msg hist should only have:
    # Bob's msg hist:
    # sys msg
    # alice ask
    # ll responds 16
    assert len(bob_hist) == 3
