from llmagent.agent.base import Agent, AgentConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.language_models.base import LLMConfig, StreamingIfAllowed
from llmagent.parsing.parser import ParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig


def test_agent():
    """
    Test whether the combined configs work as expected.
    """
    cfg = AgentConfig(
        name="test-llmagent",
        debug=False,
        vecdb=QdrantDBConfig(
            type="qdrant",
            collection_name="test",
            embedding=OpenAIEmbeddingsConfig(
                model_type="openai",
                model_name="text-embedding-ada-002",
                dims=1536,
            ),
        ),
        llm=LLMConfig(
            type="openai",
        ),
        parsing=ParsingConfig(),
        prompts=PromptsConfig(),
    )

    agent = Agent(cfg)
    response = agent.respond("what is the capital of France?")  # direct LLM question
    assert "Paris" in response.content

    with StreamingIfAllowed(agent.llm, False):
        response = agent.respond("what is the capital of France?")
    assert "Paris" in response.content
