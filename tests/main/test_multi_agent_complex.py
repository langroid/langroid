import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tools.recipient_tool import RecipientTool
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global
from langroid.vector_store.base import VectorStoreConfig


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


EXPONENTIALS = "3**4 8**3"


@pytest.mark.parametrize("fn_api", [True, False])
@pytest.mark.parametrize("constrain_recipients", [True, False])
def test_agents_with_recipient(
    test_settings: Settings,
    fn_api: bool,
    constrain_recipients: bool,
):
    set_global(test_settings)
    master_cfg = _TestChatAgentConfig(name="Master")

    planner_cfg = _TestChatAgentConfig(
        name="Planner",
        use_tools=not fn_api,
        use_functions_api=fn_api,
    )

    multiplier_cfg = _TestChatAgentConfig(name="Multiplier")

    # master asks a series of exponential questions, e.g. 3^6, 8^5, etc.
    master = ChatAgent(master_cfg)
    task_master = Task(
        master,
        interactive=False,
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
    )

    # For a given exponential computation, plans a sequence of multiplications.
    planner = ChatAgent(planner_cfg)

    if constrain_recipients:
        planner.enable_message(
            RecipientTool.create(recipients=["Master", "Multiplier"])
        )
    else:
        planner.enable_message(RecipientTool)

    task_planner = Task(
        planner,
        interactive=False,
        system_message="""
                From "Master", you will receive an exponential to compute, 
                but you do not know how to multiply. You have a helper called 
                "Multiplier" who can compute multiplications. So to calculate the
                exponential you receive from "Master", you have to ask a sequence of
                multiplication questions to "Multiplier", to figure out the 
                exponential.
                
                When you have your final answer, report your answer
                back to "Master" using the same `recipient_message` tool/function-call.
                
                When asking the Multiplier, remember to only present your 
                request in arithmetic notation, e.g. "3*5"; do not add 
                un-necessary phrases.
                """,
    )

    # Given a multiplication, returns the answer.
    multiplier = ChatAgent(multiplier_cfg)
    task_multiplier = Task(
        multiplier,
        done_if_response=[Entity.LLM],
        interactive=False,
        system_message="""
                You are a calculator. You will be given a multiplication problem. 
                You simply reply with the answer, say nothing else.
                """,
    )

    # planner helps master...
    task_master.add_sub_task(task_planner)
    # multiplier helps planner, but use Validator to ensure
    # recipient is specified via TO[recipient], and if not
    # then the validator will ask for clarification
    task_planner.add_sub_task(task_multiplier)

    result = task_master.run()

    answers = [str(eval(e)) for e in EXPONENTIALS.split()]
    assert all(a in result.content for a in answers)
    # TODO assertions on message history of each agent
