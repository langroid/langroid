#!/usr/bin/env python3
"""
Minimal test to understand the tool generation issue.
"""
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.pydantic_v1 import BaseModel, Field


class Country(BaseModel):
    """Info about a country"""

    name: str = Field(..., description="Name of the country")
    capital: str = Field(..., description="Capital of the country")


class President(BaseModel):
    """Info about a president of a country"""

    country: Country = Field(..., description="Country of the president")
    name: str = Field(..., description="Name of the president")
    election_year: int = Field(..., description="Year of election of the president")


class PresidentTool(ToolMessage):
    """Tool/function to generate a president example"""

    request: str = "show_president"
    purpose: str = """To generate an example of a president"""
    president: President = Field(..., description="An example of a president")

    def handle(self) -> str:
        return self.president.country.name


# Test configuration
cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    ),
)

# Create agent with use_functions_api=False and use_tools_api=True
agent = ChatAgent(cfg)
agent.config.use_functions_api = False
agent.config.use_tools = True  # This is set to True when use_functions_api=False
agent.config.use_tools_api = (
    True  # This should have no effect when use_functions_api=False
)
agent.enable_message(PresidentTool)

# Check what instructions are generated
print("=== System Tool Format Instructions ===")
print(agent.system_tool_format_instructions)
print("\n=== System Tool Instructions ===")
print(agent.system_tool_instructions)
print("\n=== Full System Message ===")
print(agent._create_system_and_tools_message().content)

# Check the configuration
print("\n=== Configuration ===")
print(f"use_functions_api: {agent.config.use_functions_api}")
print(f"use_tools: {agent.config.use_tools}")
print(f"use_tools_api: {agent.config.use_tools_api}")
print(f"llm_tools_usable: {agent.llm_tools_usable}")
print(f"llm_functions_usable: {agent.llm_functions_usable}")
