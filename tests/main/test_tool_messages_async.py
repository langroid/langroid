import asyncio
import itertools
import json
from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, Field

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.xml_tool_message import XMLToolMessage
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import (
    LLMFunctionCall,
    LLMFunctionSpec,
    LLMMessage,
    OpenAIJsonSchemaSpec,
    OpenAIToolCall,
    OpenAIToolSpec,
    Role,
)
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.configuration import Settings, set_global


class CountryCapitalMessage(ToolMessage):
    request: str = "country_capital"
    purpose: str = "To check whether <city> is the capital of <country>."
    country: str = "France"
    city: str = "Paris"
    result: str = "yes"  # or "no"

    @classmethod
    def examples(cls) -> List["CountryCapitalMessage"]:
        return [
            cls(country="France", city="Paris", result="yes"),
            cls(country="France", city="Marseille", result="no"),
        ]


class FileExistsMessage(ToolMessage):
    request: str = "file_exists"
    purpose: str = "To check whether a certain <filename> is in the repo."
    filename: str = Field(..., description="File name to check existence of")
    result: str = "yes"  # or "no"

    @classmethod
    def examples(cls) -> List["FileExistsMessage"]:
        return [
            cls(filename="README.md", result="yes"),
            cls(filename="Dockerfile", result="no"),
        ]


class PythonVersionMessage(ToolMessage):
    request: str = "python_version"
    _handler: str = "tool_handler"
    purpose: str = "To check which version of Python is needed."
    result: str = "3.9"

    @classmethod
    def examples(cls) -> List["PythonVersionMessage"]:
        return [
            cls(result="3.7"),
            cls(result="3.8"),
        ]


DEFAULT_PY_VERSION = "3.9"


class MessageHandlingAgent(ChatAgent):
    def file_exists(self, message: FileExistsMessage) -> str:
        return "yes" if message.filename == "requirements.txt" else "no"

    def tool_handler(self, message: ToolMessage) -> str:
        if message.request == "python_version":
            return DEFAULT_PY_VERSION
        else:
            return "invalid tool name"

    async def country_capital_async(self, message: CountryCapitalMessage) -> str:
        await asyncio.sleep(1)
        return (
            "yes" if (message.city == "Paris" and message.country == "France") else "no"
        )


cfg = ChatAgentConfig(
    name="test-langroid",
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        cache_config=RedisCacheConfig(fake=False),
    ),
    parsing=ParsingConfig(),
    prompts=PromptsConfig(),
    use_functions_api=False,
    use_tools=True,
)
agent = MessageHandlingAgent(cfg)

# Define the range of values each variable can have
use_vals = [True, False]
handle_vals = [True, False]
force_vals = [True, False]
message_classes = [None, FileExistsMessage, PythonVersionMessage]

# Get the cartesian product
cartesian_product = list(
    itertools.product(message_classes, use_vals, handle_vals, force_vals)
)

agent.enable_message(FileExistsMessage)
agent.enable_message(PythonVersionMessage)

NONE_MSG = "nothing to see here"

FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists",
"filename": "test.txt"
} 
Hope you can tell me!
"""

PYTHON_VERSION_MSG = """
great, please tell me this --
{
"request": "python_version"
}/if you know it
"""


BAD_FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists"
} 
Hope you can tell me!
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_functions_api",
    [True, False],
)
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize(
    "message_class, prompt, result",
    [
        (
            FileExistsMessage,
            "You have to find out whether the file 'requirements.txt' exists",
            "yes",
        ),
        (
            PythonVersionMessage,
            "Find out about the python version",
            "3.9",
        ),
        (
            CountryCapitalMessage,
            "You have to check whether Paris is the capital of France",
            "yes",
        ),
    ],
)
async def test_llm_tool_message(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
    message_class: ToolMessage,
    prompt: str,
    result: str,
):
    """
    Test whether LLM is able to GENERATE message (tool) in required format, and the
    agent handles the message correctly.
    Args:
        test_settings: test settings from conftest.py
        use_functions_api: whether to use LLM's functions api or not
            (i.e. use the langroid ToolMessage tools instead).
        message_class: the message class (i.e. tool/function) to test
        prompt: the prompt to use to induce the LLM to use the tool
        result: the expected result from agent handling the tool-message
    """
    set_global(test_settings)
    agent = MessageHandlingAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = use_tools_api
    agent.config.use_tools = not use_functions_api
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    agent.enable_message(CountryCapitalMessage)

    llm_msg = await agent.llm_response_forget_async(prompt)
    assert isinstance(agent.get_tool_messages(llm_msg)[0], message_class)

    agent_result = (await agent.handle_message_async(llm_msg)).content
    assert result.lower() in agent_result.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_functions_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True, False])
async def test_tool_no_llm_response_async(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    """Test that agent.llm_response does not respond to tool messages."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(CountryCapitalMessage)
    capital_tool = CountryCapitalMessage(
        city="Paris", country="France", result="yes"
    ).to_json()
    response = await agent.llm_response_async(capital_tool)
    assert response is None


# Test that malformed tool messages results in proper err msg
class NumPair(BaseModel):
    xval: int
    yval: int


class NabroskiTool(ToolMessage):
    request: str = "nabroski"
    purpose: str = "to request computing the Nabroski transform of <num_pair>"
    num_pair: NumPair

    def handle(self) -> str:
        return str(3 * self.num_pair.xval + self.num_pair.yval)


class CoriolisTool(ToolMessage):
    """Tool for testing handling of optional arguments, with default values."""

    request: str = "coriolis"
    purpose: str = "to request computing the Coriolis transform of <cats> and <cows>"
    cats: int
    cows: int = 5

    def handle(self) -> str:
        # same as NabroskiTool result
        return str(3 * self.cats + self.cows)


class NumPairE(BaseModel):
    ex: int
    ey: int


class EulerTool(ToolMessage):
    request: str = "euler"
    purpose: str = "to request computing the Euler transform of <num_paire>"
    num_paire: NumPairE

    def handle(self) -> str:
        return str(2 * self.num_paire.ex - self.num_paire.ey)


@pytest.mark.fallback
@pytest.mark.flaky(reruns=2)
@pytest.mark.asyncio
@pytest.mark.parametrize("use_fn_api", [True, False])
async def test_structured_recovery_async(use_fn_api: bool):
    """
    Test that structured fallback correctly recovers
    from failed tool calls.
    """

    async def simulate_failed_call(attempt: str | ChatDocument) -> str:
        agent = ChatAgent(
            ChatAgentConfig(
                use_functions_api=use_fn_api,
                use_tools_api=True,
                use_tools=not use_fn_api,
                strict_recovery=True,
            )
        )
        agent.enable_message(NabroskiTool)
        agent.enable_message(CoriolisTool)
        agent.enable_message(EulerTool)

        agent.message_history = [
            LLMMessage(
                role=Role.SYSTEM,
                content="You are a helpful assistant.",
            ),
            LLMMessage(
                role=Role.USER,
                content="""
                Please give me an example of a Nabroski, Coriolis, or Euler call.
                """,
            ),
            LLMMessage(
                role=Role.ASSISTANT,
                content=attempt if isinstance(attempt, str) else attempt.content,
                tool_calls=None if isinstance(attempt, str) else attempt.oai_tool_calls,
                function_call=(
                    None if isinstance(attempt, str) else attempt.function_call
                ),
            ),
        ]
        if (
            use_fn_api
            and isinstance(attempt, ChatDocument)
            and attempt.oai_tool_calls is not None
        ):
            # Inserting this since OpenAI API strictly requires a
            # Role.TOOL msg immediately after an Assistant Tool call,
            # before the next Assistant msg.
            agent.message_history.extend(
                [
                    LLMMessage(
                        role=Role.TOOL,
                        tool_call_id=t.id,
                        content="error",
                    )
                    for t in attempt.oai_tool_calls
                ]
            )

        # Simulates bad tool attempt by the LLM
        agent.handle_message(attempt)
        assert agent.tool_error
        response = await agent.llm_response_async(
            """
            There was an error in your attempted tool/function call. Please correct it.
            """
        )
        assert response is not None
        result = agent.handle_message(response)
        assert result is not None
        if isinstance(result, ChatDocument):
            return result.content

        return result

    def to_attempt(attempt: LLMFunctionCall) -> str | ChatDocument:
        if not use_fn_api:
            return json.dumps(
                {
                    "request": attempt.name,
                    **(attempt.arguments or {}),
                }
            )

        return ChatDocument(
            content="",
            metadata=ChatDocMetaData(sender=Entity.LLM),
            oai_tool_calls=[
                OpenAIToolCall(
                    id="call-1234657",
                    function=attempt,
                )
            ],
        )

    # The name of the function is incorrect:
    # The LLM should correct the request to "nabroski" in recovery
    assert (
        await simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="__nabroski__",
                    arguments={
                        "xval": 1,
                        "yval": 3,
                    },
                )
            )
        )
        == "6"
    )
    # The LLM should correct the request to "nabroski" in recovery
    assert (
        await simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="Nabroski-function",
                    arguments={
                        "xval": 2,
                        "yval": 3,
                    },
                )
            )
        )
        == "9"
    )
    # Strict fallback disables the default arguments, but the LLM
    # should infer from context. In addition, the name of the
    # function is incorrect (the LLM should infer "coriolis" in
    # recovery) and the JSON output is malformed
    assert (
        await simulate_failed_call(
            """
        request ":coriolis"
        arguments {"n_cats": 1}
        """
        )
        == "8"
    )
    # The LLM should correct the request to "coriolis" in recovery
    # The LLM should infer the default argument from context
    assert (
        await simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="Coriolis",
                    arguments={
                        "cats": 1,
                    },
                )
            )
        )
        == "8"
    )
    # The LLM should correct the request to "euler" in recovery
    assert (
        await simulate_failed_call(
            to_attempt(
                LLMFunctionCall(
                    name="EulerTool",
                    arguments={
                        "ex": 6,
                        "ey": 4,
                    },
                )
            )
        )
        == "8"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True])
@pytest.mark.parametrize("parallel_tool_calls", [True, False])
async def test_strict_fallback_async(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    parallel_tool_calls: bool,
):
    """
    Test that strict tool and structured output errors
    are handled gracefully and are disabled if errors
    are caused.
    """
    set_global(test_settings)

    class BrokenStrictSchemaAgent(ChatAgent):
        def _function_args(self) -> tuple[
            Optional[list[LLMFunctionSpec]],
            str | dict[str, str],
            Optional[list[OpenAIToolSpec]],
            Optional[dict[str, dict[str, str] | str]],
            Optional[OpenAIJsonSchemaSpec],
        ]:
            """
            Implements a broken version of the correct _function_args()
            that ensures that the generated schemas are incompatible
            with OpenAI's strict decoding implementation.

            Specifically, removes the schema edits performed by
            `format_schema_for_strict()` (e.g. setting "additionalProperties"
            to False on all objects in the JSON schema).
            """
            functions, fun_call, tools, force_tool, output_format = (
                super()._function_args()
            )

            # remove schema edits for strict
            if tools is not None:
                for t in tools:
                    name = t.function.name
                    t.function = self.llm_functions_map[name]

            if self.output_format is not None and self._json_schema_available():
                self.any_strict = True
                if issubclass(self.output_format, ToolMessage) and not issubclass(
                    self.output_format, XMLToolMessage
                ):
                    spec = self.output_format.llm_function_schema(
                        request=True,
                        defaults=self.config.output_format_include_defaults,
                    )

                    output_format = OpenAIJsonSchemaSpec(
                        strict=True,
                        function=spec,
                    )
                elif issubclass(self.output_format, BaseModel):
                    param_spec = self.output_format.schema()

                    output_format = OpenAIJsonSchemaSpec(
                        strict=True,
                        function=LLMFunctionSpec(
                            name="json_output",
                            description="Strict Json output format.",
                            parameters=param_spec,
                        ),
                    )

            return functions, fun_call, tools, force_tool, output_format

    agent = BrokenStrictSchemaAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            llm=OpenAIGPTConfig(
                parallel_tool_calls=parallel_tool_calls,
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
        )
    )
    agent.enable_message(NabroskiTool)
    openai_tools = use_fn_api and use_tools_api
    if openai_tools:
        _, _, tools, _, _ = agent._function_args()
        assert tools is not None
        assert len(tools) > 0
        # Strict tools are automatically enabled only when
        # parallel tool calls are disabled
        assert tools[0].strict == (not parallel_tool_calls)

    response = await agent.llm_response_forget_async(
        """
        What is the Nabroski transform of (1,3)? Use the
        `nabroski` tool/function.
        """
    )
    result = agent.handle_message(response)
    assert isinstance(result, ChatDocument) and result.content == "6"
    assert agent.disable_strict == (openai_tools and not parallel_tool_calls)

    agent = BrokenStrictSchemaAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            llm=OpenAIGPTConfig(
                parallel_tool_calls=parallel_tool_calls,
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
        )
    )
    structured_agent = agent[NabroskiTool]
    response = await structured_agent.llm_response_forget_async(
        """
        What is the Nabroski transform of (1,3)?
        """
    )
    assert response is not None
    assert structured_agent.disable_strict
    assert not agent.disable_strict


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("parallel_tool_calls", [True, False])
async def test_strict_schema_mismatch_async(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    parallel_tool_calls: bool,
):
    """
    Test that validation errors triggered in strict result in disabled strict ouput.
    """
    set_global(test_settings)

    def int_schema(request: str) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x": {"type": "integer"},
                "request": {"type": "string", "enum": [request]},
            },
            "required": ["x", "request"],
        }

    class WrongSchemaAgent(ChatAgent):
        def _function_args(self) -> tuple[
            Optional[List[LLMFunctionSpec]],
            str | dict[str, str],
            Optional[list[OpenAIToolSpec]],
            Optional[dict[str, dict[str, str] | str]],
            Optional[OpenAIJsonSchemaSpec],
        ]:
            """
            Implements a broken version of the correct _function_args()
            that replaces the output and all tool schemas with an
            incorrect schema. Simulates mismatched schemas due to
            schema edits.
            """
            functions, fun_call, tools, force_tool, output_format = (
                super()._function_args()
            )

            # remove schema edits for strict
            if tools is not None:
                for t in tools:
                    name = t.function.name
                    t.function.parameters = int_schema(name)

            if self.output_format is not None and self._json_schema_available():
                output_format = OpenAIJsonSchemaSpec(
                    strict=True,
                    function=LLMFunctionSpec(
                        name="json_output",
                        description="Strict Json output format.",
                        parameters=int_schema("json_output"),
                    ),
                )

            return functions, fun_call, tools, force_tool, output_format

    agent = WrongSchemaAgent(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            llm=OpenAIGPTConfig(
                parallel_tool_calls=parallel_tool_calls,
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
        )
    )

    class IntTool(ToolMessage):
        request: str = "int_tool"
        purpose: str = "To return an integer value"
        x: int

        def handle(self):
            return self.x

    class StrTool(ToolMessage):
        request: str = "str_tool"
        purpose: str = "To return an string value"
        text: str

        def handle(self):
            return self.text

    agent.enable_message(IntTool)
    agent.enable_message(StrTool)
    strict_openai_tools = use_fn_api and use_tools_api and not parallel_tool_calls
    response = await agent.llm_response_forget_async(
        """
        What is the smallest integer greater than pi? Use the
        `int_tool` tool/function.
        """
    )
    agent.handle_message(response)
    assert "int_tool" not in agent.disable_strict_tools_set

    await agent.llm_response_forget_async(
        """
        Who is the president of France? Use the `str_tool` tool/function.
        """
    )
    assert ("str_tool" in agent.disable_strict_tools_set) == strict_openai_tools

    strict_agent = agent[IntTool]
    await strict_agent.llm_response_forget_async(
        "What is the smallest integer greater than pi?"
    )
    assert not strict_agent.disable_strict

    strict_agent = agent[StrTool]
    await strict_agent.llm_response_forget_async("Who is the president of France?")
    assert strict_agent.disable_strict


class GetTimeTool(ToolMessage):
    purpose: str = "Get current time"
    request: str = "get_time"

    def response(self, agent: ChatAgent) -> ChatDocument:
        return agent.create_agent_response(
            content=json.dumps(
                {
                    "time": "11:59:59",
                    "date": "1999-12-31",
                    "day_of_week": "Friday",
                    "week_number": "52",
                    "tzname": "America/New York",
                }
            ),
            recipient=Entity.LLM,
        )


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.asyncio
async def test_strict_recovery_only_from_LLM_async(
    use_fn_api: bool,
    use_tools_api: bool,
):
    """
    Test that structured fallback only occurs on messages
    sent by the LLM.
    """
    was_tool_error = False

    class TrackToolError(ChatAgent):
        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            nonlocal was_tool_error
            if self.tool_error:
                was_tool_error = True
            return super().llm_response(message)

        async def llm_response_async(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            nonlocal was_tool_error
            if self.tool_error:
                was_tool_error = True
            return await super().llm_response_async(message)

    agent = TrackToolError(
        ChatAgentConfig(
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            strict_recovery=True,
            llm=OpenAIGPTConfig(
                supports_json_schema=True,
                supports_strict_tools=True,
            ),
            system_message="""
            You are a helpful assistant.  Start by calling the
            get_time tool. Then greet the user according to the time
            of the day.
            """,
        )
    )
    agent.enable_message(GetTimeTool)
    task = Task(agent, interactive=False)
    await task.run_async(turns=6)
    assert not was_tool_error
