from typing import List

import pytest
from pydantic import BaseModel, Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import (
    OpenAIChatModel,
    OpenAIGPTConfig,
)
from langroid.utils.configuration import Settings, set_global

cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        cache_config=RedisCacheConfig(fake=False),
    ),
)


# on purpose choose a non-descriptive name so
# we are sure the LLM is not filling in fields based on the name
# (E.g. we want to avoid names like Address).
class Blob(BaseModel):
    """Info about a blob"""

    country: str = Field(..., description="Country of origin of the blob")
    age: int = Field(..., description="Age of the blob")
    religion: str = Field(..., description="Religion of the blob")


class SuperBlob(BaseModel):
    """Info about a super-blob, who has a supername and contains a blob"""

    blob: Blob = Field(..., description="A blob")
    supername: str = Field(..., description="Name of the super-blob")


class BlobList(BaseModel):
    blobs: List[Blob] = Field(..., description="List of blobs")


class BlobListTool(ToolMessage):
    request: str = "blob_list"
    purpose: str = """To show a list of example blobs"""
    my_blobs: BlobList = Field(..., description="List of blobs")

    def handle(self) -> str:
        return str(len(self.my_blobs.blobs))

    @classmethod
    def examples(cls) -> List["BlobListTool"]:
        return [
            cls(
                my_blobs=BlobList(
                    blobs=[
                        Blob(country="USA", age=100, religion="Christian"),
                        Blob(country="China", age=20, religion="Buddhist"),
                        Blob(country="India", age=30, religion="Hindu"),
                    ]
                )
            ),
        ]


class SuperBlobTool(ToolMessage):
    request: str = "super_blob"
    purpose: str = """To generate a SuperBlob example"""
    hyper: SuperBlob = Field(..., description="A super_blob example")

    def handle(self) -> str:
        return self.hyper.blob.country

    @classmethod
    def examples(cls) -> List["SuperBlobTool"]:
        return [
            cls(
                hyper=SuperBlob(
                    supername="Superman",
                    blob=Blob(country="USA", age=100, religion="Christian"),
                )
            )
        ]


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_structured_output_list(
    test_settings: Settings,
    use_functions_api: bool,
):
    """
    Test whether LLM is able to GENERATE structured output.
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.enable_message(BlobListTool)
    N = 4
    prompt = f"Show me a list of {N} blobs, using the blob_list tool/function"
    llm_msg = agent.llm_response_forget(prompt)
    tool_name = BlobListTool.default_value("request")
    if use_functions_api:
        assert llm_msg.function_call is not None
        assert llm_msg.function_call.name == tool_name
    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], BlobListTool)

    agent_result = agent.agent_response(llm_msg)
    assert agent_result.content == str(N)


@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_structured_output_nested(
    test_settings: Settings,
    use_functions_api: bool,
):
    """
    Test whether LLM is able to GENERATE structured output.
    """
    set_global(test_settings)
    agent = ChatAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    agent.enable_message(SuperBlobTool)
    prompt = "Show me an example of a SuperBlob"
    llm_msg = agent.llm_response_forget(prompt)
    tool_name = SuperBlobTool.default_value("request")
    if use_functions_api:
        assert llm_msg.function_call is not None
        assert llm_msg.function_call.name == tool_name
    else:
        tools = agent.get_tool_messages(llm_msg)
        assert len(tools) == 1
        assert isinstance(tools[0], SuperBlobTool)

    agent.agent_response(llm_msg)
