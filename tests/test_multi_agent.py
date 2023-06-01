from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from llmagent.agent.base import Entity
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMMessage, Role
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.configuration import Settings, set_global
from llmagent.mytypes import Document, DocMetaData
from typing import Optional
import pytest


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
    agent_helper = ChatAgent(cfg2)
    agent.add_agent(agent_helper)

    agent.default_human_response = ""
    agent_helper.default_human_response = helper_human_response

    msg = """
    Your job is to ask me questions. 
    Start by asking me what the capital of France is.
    """
    agent.init_chat(user_message=msg)

    agent.process_pending_message()  # LLM asks
    assert "What" in agent.pending_message.content
    assert agent.pending_message.metadata.source == Entity.LLM
    assert agent.pending_message.content == agent.current_response.content

    agent.process_pending_message()
    # user responds '' (empty) to force agent to hand off to agent_helper,
    # and we test two possible human answers: empty or 'q'

    assert agent_helper.task_done()
    assert "Paris" in agent_helper.task_result().content
    assert "Paris" in agent.task_result().content


# The classes below are for the mult-agent test
class _MasterAgent(ChatAgent):
    def task_done(self) -> bool:
        return "DONE" in self.pending_message.content

    def task_result(self) -> Optional[Document]:
        answers = [m.content for m in self.message_history if m.role == Role.USER]
        return Document(
            content=" ".join(answers),
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
        )


class _PlannerAgent(ChatAgent):
    def task_done(self) -> bool:
        return "DONE" in self.pending_message.content

    def task_result(self) -> Optional[Document]:
        return Document(
            content=self.pending_message.content.replace("DONE:", "").strip(),
            metadata=DocMetaData(source=Entity.USER, sender=Entity.USER),
        )


class _MultiplierAgent(ChatAgent):
    def task_done(self) -> bool:
        # multiplication gets done in 1 round, so stop as soon as LLM replies
        return self.pending_message.metadata.sender == Entity.LLM


EXPONENTIALS = "3**5 8**4 9**3"


def test_multi_agent(test_settings: Settings):
    set_global(test_settings)
    master_cfg = _TestChatAgentConfig(name="Master")
    planner_cfg = _TestChatAgentConfig(name="Planner")
    multiplier_cfg = _TestChatAgentConfig(name="Multiplier")

    # master asks a series of expenenential questions, e.g. 3^6, 8^5, etc.
    master = _MasterAgent(
        master_cfg,
        task=[
            LLMMessage(
                role=Role.SYSTEM,
                content=f"""
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
            ),
            LLMMessage(
                role=Role.USER, content="Start by asking me an exponential question."
            ),
        ],
    )

    # For a given exponential computation, plans a sequence of multiplications.
    planner = _PlannerAgent(
        planner_cfg,
        task=[
            LLMMessage(
                role=Role.SYSTEM,
                content="""
                You understand exponentials, but you do not know how to multiply.
                You will be given an exponential to compute, and you have to ask a 
                sequence of multiplication questions, to figure out the exponential. 
                Present the question using only numbers, e.g, "3 * 5", and it should 
                only involve a SINGLE multiplication. 
                When you have your final answer, reply with something like 
                "DONE: 92"
                """,
            ),
        ],
    )

    # Given a multiplication, returns the answer.
    multiplier = _MultiplierAgent(
        multiplier_cfg,
        task=[
            LLMMessage(
                role=Role.SYSTEM,
                content="""
                You are a calculator. You will be given a multiplication problem. 
                You simply reply with the answer, say nothing else.
                """,
            ),
        ],
    )

    # planner helps master...
    master.add_agent(planner)
    # multiplier helps planner...
    planner.add_agent(multiplier)

    # ... since human has nothing to say
    master.default_human_response = ""
    planner.default_human_response = ""
    multiplier.default_human_response = ""

    result = master.do_task()

    answer_string = " ".join([str(eval(e)) for e in EXPONENTIALS.split()])
    assert answer_string in result.content

    # asserttions on message history of each agent
