"""Tool names survive native response conversion and agent dispatch."""

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.base import LLMFunctionCall, LLMResponse, OpenAIToolCall


def _response(call: LLMFunctionCall, tools_api: bool) -> LLMResponse:
    if tools_api:
        return LLMResponse(
            message="",
            oai_tool_calls=[OpenAIToolCall(id="call-test", function=call)],
        )
    return LLMResponse(message="", function_call=call)


@pytest.mark.parametrize("tools_api", [False, True])
@pytest.mark.parametrize(
    "tool_name",
    ["list_functions", "functions_lookup", "functions", "my_functions_tool"],
)
def test_function_names_are_preserved_for_agent_dispatch(
    tool_name: str, tools_api: bool
) -> None:
    """A valid name containing 'functions' must reach its registered handler."""

    class NamedTool(ToolMessage):
        request: str = tool_name
        purpose: str = "Return the supplied value."
        value: int

        def handle(self) -> str:
            return f"value={self.value}"

    agent = ChatAgent(ChatAgentConfig(llm=None))
    agent.enable_message(NamedTool)
    call = LLMFunctionCall(name=tool_name, arguments={"value": 42})
    document = ChatDocument.from_LLMResponse(_response(call, tools_api))

    result = agent.agent_response(document)

    assert result is not None
    assert result.content == "value=42"
    assert call.name == tool_name


@pytest.mark.parametrize("tools_api", [False, True])
@pytest.mark.parametrize(
    "name", ["functions list_functions", " functions\tlist_functions "]
)
def test_stray_function_prefix_is_still_repaired(name: str, tools_api: bool) -> None:
    """Repair only the separate prefix, keeping the actual tool name intact."""
    call = LLMFunctionCall(name=name, arguments={"value": 42})

    ChatDocument.from_LLMResponse(_response(call, tools_api))

    assert call.name == "list_functions"
    assert call.arguments == {"value": 42}


@pytest.mark.parametrize("tools_api", [False, True])
def test_explicit_request_still_overrides_function_name(tools_api: bool) -> None:
    """Preserve the existing recovery from a request inside arguments."""
    call = LLMFunctionCall(
        name="functions", arguments={"request": "list_functions", "value": 42}
    )

    ChatDocument.from_LLMResponse(_response(call, tools_api))

    assert call.name == "list_functions"
    assert call.arguments == {"value": 42}
