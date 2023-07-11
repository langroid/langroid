from typing import List, Optional

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.message import AgentMessage
from langroid.agent.special.recipient_validator_agent import (
    RecipientValidator,
    RecipientValidatorConfig,
)
from langroid.agent.task import Task
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import Role
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import DocMetaData, Document, Entity
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


class ExponentialTool(AgentMessage):
    request: str = "calc_expontential"
    purpose: str = "To calculate the value of <x> raised to the power <e>"
    x: int
    e: int
    result: int

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(x=3, e=5, result=243),
            cls(x=8, e=3, result=512),
        ]


class MultiplicationTool(AgentMessage):
    request: str = "calc_multiplication"
    purpose: str = "To calculate the value of <x> multiplied by <y>"
    x: int
    y: int
    result: int

    @classmethod
    def examples(cls) -> List["AgentMessage"]:
        return [
            cls(x=3, y=5, result=15),
            cls(x=8, y=3, result=24),
        ]


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


def test_agents_with_validator(test_settings: Settings):
    set_global(test_settings)
    master_cfg = _TestChatAgentConfig(name="Master")

    planner_cfg = _TestChatAgentConfig(name="Planner")

    multiplier_cfg = _TestChatAgentConfig(name="Multiplier")

    val_cfg = RecipientValidatorConfig(
        llm=None,
        vecdb=None,
        name="Validator",
        recipients=["Master", "Multiplier"],
        tool_recipient="Multiplier",
    )

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
                From "Master", you will receive an exponential to compute, 
                but you do not know how to multiply. You have a helper called 
                "Multiplier" who can compute multiplications. So to calculate the
                exponential you receive from "Master", you have to ask a sequence of
                multiplication questions to "Multiplier", to figure out the 
                exponential. You must ask the Multiplier in the format
                "TO[Multiplier]: 3 * 5", and it should only involve a SINGLE 
                multiplication. When you have your final answer, report your answer
                back to "Master" in the format "TO[Master]: 243". 
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

    validator = RecipientValidator(val_cfg)
    task_validator = Task(
        validator,
        name="Validator",
        single_round=True,
    )
    # planner helps master...
    task_master.add_sub_task(task_planner)
    # multiplier helps planner, but use Validator to ensure
    # recipient is specified via TO[recipient], and if not
    # then the validator will ask for clarification
    task_planner.add_sub_task([task_validator, task_multiplier])

    # ... since human has nothing to say
    master.default_human_response = ""
    planner.default_human_response = ""
    multiplier.default_human_response = ""

    result = task_master.run()

    answers = [str(eval(e)) for e in EXPONENTIALS.split()]
    assert all(a in result.content for a in answers)
    # TODO assertions on message history of each agent
