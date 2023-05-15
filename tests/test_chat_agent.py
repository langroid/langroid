from llmagent.agent.chat_agent import ChatAgent
from llmagent.language_models.base import LLMMessage, Role, StreamingIfAllowed
from llmagent.agent.base import AgentConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig


class COTAgentConfig(AgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: LLMConfig = LLMConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=True),
    )
    parsing: ParsingConfig = None
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


def test_cot_agent():
    cfg = COTAgentConfig()
    task = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""you are a devops engineer. 
                   You have access to a repo. You have to write a docker file to 
                   containerize it. Come up with a plan to do this, 
                   breaking it down into small steps. 
                   You have to ask me what you need to know in order to complete 
                   your task, ONE STEP AT A TIME. 
                   At any time you can ONLY ASK me a SINGLE QUESTION.
                   I will answer it, and then you will ASK another QUESTION,
                   and son on, until you say you are DONE, and show me the completed 
                   dockerfile.""",
        ),
        LLMMessage(
            role=Role.SYSTEM,
            name="example_user",
            content="Please send me your first QUESTION",
        ),
        LLMMessage(
            role=Role.SYSTEM,
            name="example_assistant",
            content="What language is the repo written in?",
        ),
        LLMMessage(role=Role.SYSTEM, name="example_user", content="Python"),
        LLMMessage(
            role=Role.SYSTEM,
            name="example_assistant",
            content="What version of Python?",
        ),
        LLMMessage(role=Role.SYSTEM, name="example_user", content="3.8"),
        LLMMessage(
            role=Role.USER,
            content="""You are a sophisticated devops engineer, and you 
                           need to write a dockerfile for a repo. Please think 
                           in small steps, and do this gradually, and focus on what 
                           information you need to know to accomplish your task.
                           Please send me your first QUESTION""",
        ),
    ]
    # just testing that these don't fail
    agent = ChatAgent(cfg, task)
    agent.start()
    agent.respond("I am not sure")

    with StreamingIfAllowed(agent.llm, False):
        agent.respond("how are you?")
