from typing import Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.message import AgentMessage
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import Role
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER
from langroid.vector_store.base import VectorStoreConfig


class ExponentialTool(AgentMessage):
    request: str = "calc_expontential"
    purpose: str = "To calculate the value of <x> raised to the power <e>"
    x: int
    e: int


class MultiplicationTool(AgentMessage):
    request: str = "calc_multiplication"
    purpose: str = "To calculate the value of <x> multiplied by <y>"
    x: int
    y: int


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
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
        llm_delegate=True,
        single_round=False,
        default_human_response="",
        only_user_quits_root=False,
    )
    agent_helper = ChatAgent(cfg2)
    task_helper = Task(
        agent_helper,
        llm_delegate=False,
        single_round=True,
        default_human_response=helper_human_response,
    )
    task.add_sub_task(task_helper)

    msg = """
    Your job is to ask me questions. 
    Start by asking me what the capital of France is.
    """
    task.init_pending_message(msg)

    task.step()
    assert "What" in task.pending_message.content
    assert task.pending_message.metadata.source == Entity.LLM

    task.step()
    # user responds '' (empty) to force agent to hand off to agent_helper,
    # and we test two possible human answers: empty or 'q'

    assert "Paris" in task_helper.result().content


# The classes below are for the mult-agent test
class _MasterAgent(ChatAgent):
    def _task_done(self) -> bool:
        return "DONE" in self.pending_message.content

    def task_result(self) -> Optional[Document]:
        answers = [m.content for m in self.message_history if m.role == Role.USER]
        return Document(
            content=" ".join(answers),
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
        )


class _PlannerAgent(ChatAgent):
    def _task_done(self) -> bool:
        return "DONE" in self.pending_message.content

    def task_result(self) -> Optional[Document]:
        return Document(
            content=self.pending_message.content.replace("DONE:", "").strip(),
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
        )


class _MultiplierAgent(ChatAgent):
    def _task_done(self) -> bool:
        # multiplication gets done in 1 round, so stop as soon as LLM replies
        return self.pending_message.metadata.sender == Entity.LLM


EXPONENTIALS = "3**5 8**3 9**3"


def test_multi_agent(test_settings: Settings):
    set_global(test_settings)
    master_cfg = _TestChatAgentConfig(name="Master")

    planner_cfg = _TestChatAgentConfig(name="Planner")

    multiplier_cfg = _TestChatAgentConfig(name="Multiplier")

    # master asks a series of expenenential questions, e.g. 3^6, 8^5, etc.
    master = _MasterAgent(master_cfg)
    task_master = Task(
        master,
        llm_delegate=True,
        single_round=False,
        default_human_response="",
        system_message=f"""
                Your job is to ask me EXACTLY this series of exponential questions:
                {EXPONENTIALS}
                Simply present the needed computation, one at a time, 
                using only numbers and the exponential operator "**".
                Say nothing else, only the numerical operation.
                When you receive the answer, say RIGHT or WRONG, and ask 
                the next exponential question, e.g.: "RIGHT 8**2".
                When done asking the series of questions, simply 
                say "DONE:" followed by the answers without commas, 
                e.g. "DONE: 243 512 729 125".
                """,
        user_message="Start by asking me an exponential question.",
        only_user_quits_root=False,
    )

    # For a given exponential computation, plans a sequence of multiplications.
    planner = _PlannerAgent(planner_cfg)
    task_planner = Task(
        planner,
        llm_delegate=True,
        single_round=False,
        default_human_response="",
        system_message="""
                You understand exponentials, but you do not know how to multiply.
                You will be given an exponential to compute, and you have to ask a 
                sequence of multiplication questions, to figure out the exponential. 
                Present the question using only numbers, e.g, "3 * 5", and it should 
                only involve a SINGLE multiplication. 
                When you have your final answer, reply with something like 
                "DONE: 92"
                """,
    )

    # Given a multiplication, returns the answer.
    multiplier = _MultiplierAgent(multiplier_cfg)
    task_multiplier = Task(
        multiplier,
        llm_delegate=False,
        single_round=True,
        default_human_response="",
        system_message="""
                You are a calculator. You will be given a multiplication problem. 
                You simply reply with the answer, say nothing else.
                """,
    )

    # planner helps master...
    task_master.add_sub_task(task_planner)
    # multiplier helps planner...
    task_planner.add_sub_task(task_multiplier)

    # ... since human has nothing to say
    master.default_human_response = ""
    planner.default_human_response = ""
    multiplier.default_human_response = ""

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
        default_human_response="",
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
        default_human_response="",
        single_round=True,
    )

    task_c = Task(
        agent_c,
        system_message=f"your job is to always say '{C_RESPONSE}'",
        default_human_response="",
        single_round=True,
    )

    task_a.add_sub_task([task_b, task_c])
    # kick off with empty msg, so LLM will respond based on initial sys, user messages
    task_a.init_pending_message()
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
    Test whether TO:[<recipient>] works as expected.
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
        default_human_response="",
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
    task_b = Task(
        agent_b,
        system_message=f"your job is to always say '{NO_ANSWER}'",
        default_human_response="",
        single_round=True,
    )

    task_c = Task(
        agent_c,
        system_message=f"your job is to always say '{NO_ANSWER}'",
        default_human_response="",
        single_round=True,
    )

    task_a.add_sub_task([task_b, task_c])
    # kick off with empty msg, so LLM will respond based on initial sys, user messages
    task_a.init_pending_message()
    for _ in range(2):
        # LLM asks, addressing B or C
        task_a.step()
        # recipient replies NO_ANSWER
        task_a.step()
        assert NO_ANSWER in task_a.pending_message.content

    task_a.agent.clear_history(0)
    result = task_a.run(turns=2)
    assert NO_ANSWER in result.content
