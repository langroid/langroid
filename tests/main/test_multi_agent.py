from typing import Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import StatusCode
from langroid.agent.task import Task
from langroid.agent.tools.orchestration import DoneTool
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE, NO_ANSWER
from langroid.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: Optional[VectorStoreConfig] = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


@pytest.mark.parametrize("helper_human_response", ["", "q"])
def test_inter_agent_chat(test_settings: Settings, helper_human_response: str):
    set_global(test_settings)
    cfg1 = _TestChatAgentConfig(name="master")
    cfg2 = _TestChatAgentConfig(name="helper")

    agent = ChatAgent(cfg1)
    task = Task(
        agent,
        interactive=False,
    )
    agent_helper = ChatAgent(cfg2)
    task_helper = Task(
        agent_helper,
        done_if_no_response=[Entity.LLM],
        done_if_response=[Entity.LLM],
        default_human_response=helper_human_response,
    )
    task.add_sub_task(task_helper)

    msg = """
    Your job is to ask me questions. 
    Start by asking me what the capital of France is.
    """
    task.init(msg)

    task.step()
    assert "What" in task.pending_message.content
    assert task.pending_message.metadata.source == Entity.LLM

    task.step()
    # user responds '' (empty) to force agent to hand off to agent_helper,
    # and we test two possible human answers: empty or 'q'

    assert "Paris" in task_helper.result().content


EXPONENTIALS = "3**5 8**3 9**3"


@pytest.mark.parametrize("use_done_tool", [True, False])
def test_multi_agent(test_settings: Settings, use_done_tool: bool):
    set_global(test_settings)
    master_cfg = _TestChatAgentConfig(name="Master")

    planner_cfg = _TestChatAgentConfig(name="Planner")

    multiplier_cfg = _TestChatAgentConfig(name="Multiplier")

    # master asks a series of exponential questions, e.g. 3^6, 8^5, etc.
    master = ChatAgent(master_cfg)
    master.enable_message(DoneTool)
    done_tool_name = DoneTool.default_value("request")
    if use_done_tool:
        done_response = f"""
        use the TOOL: `{done_tool_name}` with `content` field 
        equal to a string containing the answers as a SEQUENCE without commas, 
        e.g. "1000 8 64"
        """
    else:
        done_response = f"""
        say {DONE}  followed by the sequence of answers without commas,
        e.g. "{DONE}: 1000 8 64"
        """

    task_master = Task(
        master,
        interactive=False,
        system_message=f"""
                Your job is to ask  EXACTLY this series of exponential questions:
                {EXPONENTIALS}
                Simply present the needed computation, one at a time, 
                using only numbers and the exponential operator "**".
                Say nothing else, only the numerical operation.
                When you receive the answer, ask 
                the NEXT exponential question, e.g.: "8**2".
                When done asking the series of questions, 
                {done_response}
                
                EXAMPLE:
                Suppose you were told to ask these exponential questions:
                "5**3 10**4  1**5"
                
                1. you ask "5**3"
                2. you receive answer "125"
                3. You say "10**4"  <--- you are asking the NEXT EXPONENTIAL
                4. you receive answer "10000"
                5. You say "1**5"  <--- you are asking the NEXT EXPONENTIAL
                6. you receive answer "1"
                7. you use the `{done_tool_name}` TOOL to send "125 10000 1"
                     as the `content` field in the TOOL
                   
                 
                """,
        user_message="Start by asking me an exponential question.",
    )

    # For a given exponential computation, plans a sequence of multiplications.
    planner = ChatAgent(planner_cfg)
    planner.enable_message(DoneTool)

    task_planner = Task(
        planner,
        interactive=False,
        system_message=f"""
                You understand EXPONENTIALS, and you know an exponential involving
                INTEGERS is simply a sequence of MULTIPLICATIONS.
                However you do NOT know how to MULTIPLY, so you have to BREAK DOWN
                into a series of multiplications, and for each 
                multiplication, send out the desired multiplication question,
                e.g. "16 * 4", and a MULTIPLICATION EXPERT will return the
                answer to you. Then you can ask the next multiplication question,
                and so on, until you have the final answer for the original
                EXPONENTIAL question.

                When you have your final answer, use the TOOL: `{done_tool_name}`
                with content equal to the answer as a string, e.g. "256".
                
                EXAMPLE:
                1. User sends you *10 ** 3".
                2. you say "10 * 10"
                3. Multiplication expert returns 100
                4. you say "100 * 10"
                5. Multiplication expert returns 1000
                6. you have the final answer, so you 
                   use the `{done_tool_name}` TOOL to send "1000" as the `content`
                """,
    )

    # Given a multiplication, returns the answer.
    multiplier = ChatAgent(multiplier_cfg)
    task_multiplier = Task(
        multiplier,
        interactive=False,
        done_if_response=[Entity.LLM],
        system_message="""
                You are a calculator. You will be given a multiplication problem. 
                You simply reply with the answer, say nothing else.
                """,
    )

    # planner helps master...
    task_master.add_sub_task(task_planner)
    # multiplier helps planner...
    task_planner.add_sub_task(task_multiplier)

    result = task_master.run()

    answers = [str(eval(e)) for e in EXPONENTIALS.split()]
    assert all(a in result.content for a in answers)
    # TODO assertions on message history of each agent


def test_multi_agent_directed(test_settings: Settings):
    """
    Test whether TO:[<recipient>] works as expected.
    """
    set_global(test_settings)
    cfg_a = _TestChatAgentConfig(name="A")

    cfg_b = _TestChatAgentConfig(name="B")

    cfg_c = _TestChatAgentConfig(name="C")

    agent_a = ChatAgent(cfg_a)
    agent_b = ChatAgent(cfg_b)
    agent_c = ChatAgent(cfg_c)

    task_a = Task(
        agent_a,
        interactive=False,
        system_message="""
        You are talking to two people B and C, and 
        your job is to pick B or C and ask that person 'Who are you?'.
        Whoever you address, make sure you say it in the form 
        TO[<recipient>]: <your message>.
        As the conversation progresses your job is always keep asking 
        this question to either B or C.
        """,
        user_message="Start by asking B or C 'Who are you?'",
    )
    B_RESPONSE = "hello I am B"
    C_RESPONSE = "hello I am C"
    task_b = Task(
        agent_b,
        system_message=f"your job is to always say '{B_RESPONSE}'",
        interactive=False,
        done_if_no_response=[Entity.LLM],
        done_if_response=[Entity.LLM],
    )

    task_c = Task(
        agent_c,
        system_message=f"your job is to always say '{C_RESPONSE}'",
        interactive=False,
        done_if_response=[Entity.LLM],
    )

    task_a.add_sub_task([task_b, task_c])
    # kick off with empty msg, so LLM will respond based on initial sys, user messages
    task_a.init()
    for _ in range(2):
        # LLM asks, addressing B or C
        task_a.step()
        recipient = task_a.pending_message.metadata.recipient
        # recipient replies
        task_a.step()
        assert recipient in task_a.pending_message.content

    task_a.agent.clear_history(0)
    result = task_a.run(turns=2)
    assert "B" in result.content or "C" in result.content


def test_multi_agent_no_answer(test_settings: Settings):
    """
    Test whether @[<recipient>] works as expected.
    Also verfies that when LLM of subtask returns NO_ANSWER,
    the appropriate result is received by the parent task.
    """
    set_global(test_settings)
    cfg_a = _TestChatAgentConfig(name="A")

    cfg_b = _TestChatAgentConfig(name="B")

    cfg_c = _TestChatAgentConfig(name="C")

    agent_a = ChatAgent(cfg_a)
    agent_b = ChatAgent(cfg_b)
    agent_c = ChatAgent(cfg_c)

    task_a = Task(
        agent_a,
        interactive=False,
        system_message="""
        You are talking to two people B and C, and 
        your job is to pick B or C and ask that person 'Who are you?'.
        Whoever you address, make sure you say it in the form 
        @[recipient]: <your message>.
        As the conversation progresses your job is always keep asking 
        this question to either B or C.
        """,
        user_message="Start by asking B or C 'Who are you?'",
    )
    task_b = Task(
        agent_b,
        system_message=f"your job is to always say '{NO_ANSWER}'",
        interactive=False,
        done_if_response=[Entity.LLM],
    )

    task_c = Task(
        agent_c,
        system_message=f"your job is to always say '{NO_ANSWER}'",
        interactive=False,
        done_if_response=[Entity.LLM],
    )

    task_a.add_sub_task([task_b, task_c])
    # kick off with empty msg, so LLM will respond based on initial sys, user messages
    task_a.init()
    # LLM asks "Who are you", addressing B or C
    pending_message = task_a.step()
    assert "who" in pending_message.content.lower()
    assert pending_message.metadata.sender == Entity.LLM
    # recipient replies NO_ANSWER, which is considered invalid, hence
    # pending message does not change
    pending_message = task_a.step()
    assert NO_ANSWER in pending_message.content
    assert pending_message.metadata.sender == Entity.USER

    task_a.agent.clear_history(0)
    # Run for 2 turns -- recipients say NO_ANSWER, which is
    # normally an invalid response, but since this is the ONLY explicit response
    # in the step, we process this as a valid step result, and the pending message
    # is updated to this message.
    result = task_a.run(turns=2)
    assert NO_ANSWER in result.content
    assert result.metadata.status == StatusCode.FIXED_TURNS
