from typing import Optional

import pytest

from llmagent.agent.base import Entity
from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.agent.task import Task
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.language_models.base import Role
from llmagent.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from llmagent.mytypes import DocMetaData, Document
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.utils.configuration import Settings, set_global
from llmagent.vector_store.base import VectorStoreConfig


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
    cfg1 = _TestChatAgentConfig(name="Smith")
    cfg2 = _TestChatAgentConfig(name="Jones")

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
    task.reset_pending_message(msg)

    task.step()
    assert "What" in task.pending_message.content
    assert task.pending_message.metadata.source == Entity.LLM

    task.step()
    # user responds '' (empty) to force agent to hand off to agent_helper,
    # and we test two possible human answers: empty or 'q'

    assert task_helper.done()
    assert "Paris" in task_helper.result().content
    assert not task.done()


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
        single_round=True,
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
