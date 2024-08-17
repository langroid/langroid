import itertools
import json
import random
from typing import Any, List, Literal, Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.task import Task
from langroid.agent.tool_message import FinalResultTool, ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.mytypes import Entity
from langroid.parsing.parse_json import extract_top_level_json
from langroid.parsing.parser import ParsingConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.pydantic_v1 import BaseModel, Field
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE


class CountryCapitalMessage(ToolMessage):
    request: str = "country_capital"
    purpose: str = "To check whether <city> is the capital of <country>."
    country: str = "France"
    city: str = "Paris"

    @classmethod
    def examples(cls) -> List["CountryCapitalMessage"]:
        # illustrating two types of examples
        return [
            (
                "Need to check if Paris is the capital of France",
                cls(country="France", city="Paris"),
            ),
            cls(country="France", city="Marseille"),
        ]


class FileExistsMessage(ToolMessage):
    request: str = "file_exists"
    purpose: str = "To check whether a certain <filename> is in the repo."
    filename: str = Field(..., description="File name to check existence of")

    @classmethod
    def examples(cls) -> List["FileExistsMessage"]:
        return [
            cls(filename="README.md"),
            cls(filename="Dockerfile"),
        ]


class PythonVersionMessage(ToolMessage):
    request: str = "python_version"
    purpose: str = "To check which version of Python is needed."

    @classmethod
    def examples(cls) -> List["PythonVersionMessage"]:
        return [
            cls(),
        ]


DEFAULT_PY_VERSION = "3.9"


class MessageHandlingAgent(ChatAgent):
    def file_exists(self, message: FileExistsMessage) -> str:
        return "yes" if message.filename == "requirements.txt" else "no"

    def python_version(self, PythonVersionMessage) -> str:
        return DEFAULT_PY_VERSION

    def country_capital(self, message: CountryCapitalMessage) -> str:
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
    system_message="""
    VERY IMPORTANT: IF you see a possibility of using a tool/function,
    you MUST use it, and MUST NOT ASK IN NATURAL LANGUAGE.
    """,
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


@pytest.mark.parametrize(
    # cartesian product of all combinations of use, handle, force
    "msg_class, use, handle, force",
    cartesian_product,
)
def test_enable_message(
    msg_class: Optional[ToolMessage], use: bool, handle: bool, force: bool
):
    agent.enable_message(msg_class, use=use, handle=handle, force=force)
    tools = agent._get_tool_list(msg_class)
    for tool in tools:
        assert tool in agent.llm_tools_map
        if msg_class is not None:
            assert agent.llm_tools_map[tool] == msg_class
            assert agent.llm_functions_map[tool] == msg_class.llm_function_schema()
        assert (tool in agent.llm_tools_handled) == handle
        assert (tool in agent.llm_tools_usable) == use
        assert (tool in agent.llm_functions_handled) == handle
        assert (tool in agent.llm_functions_usable) == use

    if msg_class is not None:
        assert (
            agent.llm_function_force is not None
            and agent.llm_function_force["name"] == tools[0]
        ) == force


@pytest.mark.parametrize("msg_class", [None, FileExistsMessage, PythonVersionMessage])
def test_disable_message_handling(msg_class: Optional[ToolMessage]):
    agent.enable_message([FileExistsMessage, PythonVersionMessage])
    usable_tools = agent.llm_tools_usable

    agent.disable_message_handling(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in set(tools).intersection(usable_tools):
        assert tool not in agent.llm_tools_handled
        assert tool not in agent.llm_functions_handled
        assert tool in agent.llm_tools_usable
        assert tool in agent.llm_functions_usable


@pytest.mark.parametrize("msg_class", [None, FileExistsMessage, PythonVersionMessage])
def test_disable_message_use(msg_class: Optional[ToolMessage]):
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    usable_tools = agent.llm_tools_usable

    agent.disable_message_use(msg_class)
    tools = agent._get_tool_list(msg_class)
    for tool in set(tools).intersection(usable_tools):
        assert tool not in agent.llm_tools_usable
        assert tool not in agent.llm_functions_usable
        assert tool in agent.llm_tools_handled
        assert tool in agent.llm_functions_handled


@pytest.mark.parametrize("msg_cls", [PythonVersionMessage, FileExistsMessage])
def test_usage_instruction(msg_cls: ToolMessage):
    usage = msg_cls.usage_examples()
    jsons = extract_top_level_json(usage)
    assert all(
        json.loads(j)["request"] == msg_cls.default_value("request") for j in jsons
    )


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


def test_agent_handle_message():
    """
    Test whether messages are handled correctly, and that
    message enabling/disabling works as expected.
    """
    agent.enable_message(FileExistsMessage)
    agent.enable_message(PythonVersionMessage)
    assert agent.handle_message(NONE_MSG) is None
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"

    agent.disable_message_handling(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"

    agent.disable_message_handling(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) is None
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) is None

    agent.enable_message(PythonVersionMessage)
    assert agent.handle_message(FILE_EXISTS_MSG) == "no"
    assert agent.handle_message(PYTHON_VERSION_MSG) == "3.9"


BAD_FILE_EXISTS_MSG = """
Ok, thank you.
{
"request": "file_exists"
} 
Hope you can tell me!
"""


def test_handle_bad_tool_message():
    """
    Test that a correct tool name with bad/missing args is
            handled correctly, i.e. the agent returns a clear
            error message to the LLM so it can try to fix it.
    """
    agent.enable_message(FileExistsMessage)
    assert agent.handle_message(NONE_MSG) is None
    result = agent.handle_message(BAD_FILE_EXISTS_MSG)
    assert "file_exists" in result and "filename" in result and "required" in result


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "use_functions_api",
    [False],
)
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
def test_llm_tool_message(
    test_settings: Settings,
    use_functions_api: bool,
    message_class: ToolMessage,
    prompt: str,
    result: str,
    stream: bool,
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
    cfg.llm.stream = stream
    agent = MessageHandlingAgent(cfg)
    agent.config.use_functions_api = use_functions_api
    agent.config.use_tools = not use_functions_api
    if not agent.llm.is_openai_chat_model() and use_functions_api:
        pytest.skip(
            f"""
            Function Calling not available for {agent.config.llm.chat_model}: skipping
            """
        )

    agent.enable_message(
        [
            FileExistsMessage,
            PythonVersionMessage,
            CountryCapitalMessage,
        ]
    )

    llm_msg = agent.llm_response_forget(prompt)
    tool_name = message_class.default_value("request")
    if use_functions_api:
        assert llm_msg.oai_tool_calls[0].function.name == tool_name

    tools = agent.get_tool_messages(llm_msg)
    assert len(tools) == 1
    assert isinstance(tools[0], message_class)

    agent_result = agent.handle_message(llm_msg)

    assert result.lower() in agent_result.lower()


def test_llm_non_tool(test_settings: Settings):
    """Having no tools enabled should result in a None handle_message result"""
    agent = MessageHandlingAgent(cfg)
    llm_msg = agent.llm_response_forget(
        "Ask me to check what is the population of France."
    ).content
    agent_result = agent.handle_message(llm_msg)
    assert agent_result is None


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
    purpose: str = "to request computing the Coriolis transform of <x> and <y>"
    x: int
    y: int = 5

    def handle(self) -> str:
        # same as NabroskiTool result
        return str(3 * self.x + self.y)


wrong_nabroski_tool = """
{
"request": "nabroski",
"num_pair": {
    "xval": 1
    }
}
"""


@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_agent_malformed_tool(
    test_settings: Settings, use_tools_api: bool, use_functions_api: bool, stream: bool
):
    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    cfg.llm.stream = stream
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    response = agent.agent_response(wrong_nabroski_tool)
    # We expect an error msg containing certain specific field names
    assert "num_pair" in response.content and "yval" in response.content


class EulerTool(ToolMessage):
    request: str = "euler"
    purpose: str = "to request computing the Euler transform of <num_pair>"
    num_pair: NumPair

    def handle(self) -> str:
        return str(2 * self.num_pair.xval - self.num_pair.yval)


class SumTool(ToolMessage):
    request: str = "sum"
    purpose: str = "to request computing the sum of <x> and <y>"
    x: int
    y: int

    def handle(self) -> str:
        return str(self.x + self.y)


class GaussTool(ToolMessage):
    request: str = "gauss"
    purpose: str = "to request computing the Gauss transform of (<x>, <y>)"
    xval: int
    yval: int

    def handle(self) -> str:
        return str((self.xval + self.yval) * self.yval)


class CoinFlipTool(ToolMessage):
    request: str = "coin_flip"
    purpose: str = "to request a random coin flip"

    def handle(self) -> Literal["Heads", "Tails"]:
        heads = random.random() > 0.5
        return "Heads" if heads else "Tails"


@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_agent_infer_tool(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    set_global(test_settings)
    gauss_request = """{"xval": 1, "yval": 3}"""
    nabrowski_or_euler_request = """{"num_pair": {"xval": 1, "yval": 3}}"""
    euler_request = """{"request": "euler", "num_pair": {"xval": 1, "yval": 3}}"""
    additional_args_request = """{"xval": 1, "yval": 3, "zval": 4}"""
    additional_args_request_specified = """
    {"request": "gauss", "xval": 1, "yval": 3, "zval": 4}
    """
    no_args_request = """{}"""
    no_args_request_specified = """{"request": "coin_flip"}"""

    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(
        [
            NabroskiTool,
            GaussTool,
            CoinFlipTool,
        ]
    )
    agent.enable_message(EulerTool, handle=False)

    # Nabrowski is the only option prior to enabling EulerTool handling
    assert agent.agent_response(nabrowski_or_euler_request).content == "6"

    # Enable handling EulerTool, this makes nabrowski_or_euler_request ambiguous
    agent.enable_message(EulerTool)

    # Gauss is the only option
    assert agent.agent_response(gauss_request).content == "12"

    # Explicit requests are forwarded to the correct handler
    assert agent.agent_response(euler_request).content == "-1"

    # We cannot infer the correct tool if there exist multiple matches
    assert agent.agent_response(nabrowski_or_euler_request) is None

    # We do not infer tools where the request has additional arguments
    assert agent.agent_response(additional_args_request) is None
    # But additional args are acceptable when the tool is specified
    assert agent.agent_response(additional_args_request_specified).content == "12"

    # We do not infer tools with no args
    assert agent.agent_response(no_args_request) is None
    # Request must be specified
    assert agent.agent_response(no_args_request_specified).content in ["Heads", "Tails"]


@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_no_llm_response(
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
    agent.enable_message(NabroskiTool)
    nabroski_tool = NabroskiTool(num_pair=NumPair(xval=1, yval=2)).to_json()
    response = agent.llm_response(nabroski_tool)
    assert response is None


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_no_task(
    test_settings: Settings,
    use_functions_api: bool,
    stream: bool,
):
    """Test tool handling without running task, i.e. directly using
    agent.llm_response and agent.agent_response methods."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
    )
    cfg.llm.stream = stream
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool, use=True, handle=True)

    response = agent.llm_response("What is Nabroski of 1 and 2?")
    assert isinstance(agent.get_tool_messages(response)[0], NabroskiTool)
    result = agent.agent_response(response)
    assert result.content == "5"


@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_tool_optional_args(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
):
    """Test that ToolMessage where some args are optional (i.e. have default values)
    works well, i.e. LLM is able to generate all args if needed, including optionals."""

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
    )
    agent = ChatAgent(cfg)

    agent.enable_message(CoriolisTool, use=True, handle=True)
    response = agent.llm_response("What is the Coriolis transform of 1, 2?")
    assert isinstance(agent.get_tool_messages(response)[0], CoriolisTool)
    tool = agent.get_tool_messages(response)[0]
    assert tool.x == 1 and tool.y == 2


@pytest.mark.parametrize("tool", [NabroskiTool, CoriolisTool])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_llm_tool_task(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
    stream: bool,
    tool: ToolMessage,
):
    """
    Test "full life cycle" of tool, when using Task.run().

    1. invoke LLM api with tool-spec
    2. LLM generates tool
    3. ChatAgent.agent_response handles tool, result added to ChatAgent msg history
    5. invoke LLM api with tool result
    """

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
        system_message=f"""
        You will be asked to compute a certain transform of two numbers, 
        using a tool/function-call that you have access to. 
        When you receive the answer from the tool, say {DONE} and show the answer.
        """,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(tool)
    task = Task(agent, interactive=False)

    request = tool.default_value("request")
    result = task.run(f"What is the {request} transform of 3 and 5?")
    assert "14" in result.content


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_functions_api", [True, False])
def test_multi_tool(
    test_settings: Settings,
    use_functions_api: bool,
    use_tools_api: bool,
    stream: bool,
):
    """
    Test "full life cycle" of tool, when using Task.run().

    1. invoke LLM api with tool-spec
    2. LLM generates tool
    3. ChatAgent.agent_response handles tool, result added to ChatAgent msg history
    5. invoke LLM api with tool result
    """

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=not use_functions_api,
        use_functions_api=use_functions_api,
        use_tools_api=use_tools_api,
        system_message=f"""
        You will be asked to compute transforms of two numbers, 
        using tools/function-calls that you have access to.
        When you are asked for MULTIPLE transforms, you MUST 
        use MULTIPLE tools/functions.  
        When you receive the answers from the tools, say {DONE} and show the answers.
        """,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(NabroskiTool)
    agent.enable_message(GaussTool)
    task = Task(agent, interactive=False)

    # First test without task; using individual methods
    # ---

    result = task.run(
        """
        Compute these: 
        (A) Nabroski transform of 3 and 5
        (B) Gauss transform of 1 and 2
        """
    )
    # Nabroski: 3*3 + 5 = 14
    # Gauss: (1+2)*2 = 6
    assert "14" in result.content and "6" in result.content


@pytest.mark.parametrize("stream", [False, True])
def test_oai_tool_choice(
    test_settings: Settings,
    stream: bool,
):
    """
    Test tool_choice for OpenAI-like LLM APIs.
    """

    set_global(test_settings)
    cfg = ChatAgentConfig(
        use_tools=False,  # langroid tools
        use_functions_api=True,  # openai tools/fns
        use_tools_api=True,  # openai tools/fns
        system_message=f"""
        You will be asked to compute an operation or transform of two numbers, 
        either using your own knowledge, or 
        using a tool/function-call that you have access to.        
        When you have an answer, say {DONE} and show the answer.
        """,
    )
    agent = ChatAgent(cfg)
    agent.enable_message(SumTool)

    chat_doc = agent.create_user_response("What is the sum of 3 and 5?")
    chat_doc.oai_tool_choice = "auto"
    response = agent.llm_response_forget(chat_doc)

    # expect either SumTool or direct result without tool
    assert "8" in response.content or isinstance(
        agent.get_tool_messages(response)[0], SumTool
    )

    chat_doc = agent.create_user_response("What is the double of 5?")
    chat_doc.oai_tool_choice = "none"
    response = agent.llm_response_forget(chat_doc)
    assert "10" in response.content

    chat_doc = agent.create_user_response("What is the sum of 3 and 5?")
    chat_doc.oai_tool_choice = "required"
    response = agent.llm_response_forget(chat_doc)
    assert isinstance(agent.get_tool_messages(response)[0], SumTool)

    agent.enable_message(NabroskiTool, force=True)
    response = agent.llm_response("What is the nabroski of 3 and 5?")
    assert "nabroski" in response.content.lower() or isinstance(
        agent.get_tool_messages(response)[0], NabroskiTool
    )


@pytest.mark.parametrize(
    "result_type",
    [
        "final_tool",
        "agent_done",
        "tool",
        "int",
        "list",
        "dict",
        "ChatDocument",
        "pydantic",
    ],
)
@pytest.mark.parametrize("tool_handler", ["handle", "response", "response_with_doc"])
def test_tool_handlers_and_results(result_type: str, tool_handler: str):
    """Test various types of ToolMessage handlers, and check that they can
    return arbitrary result types"""

    class SpecialResult(BaseModel):
        """To illustrating returning an arbitrary Pydantic object as a result"""

        answer: int
        details: str = "nothing"

    def result_fn(x: int) -> Any:
        match result_type:
            case "int":
                return x + 5
            case "dict":
                return {"answer": x + 5, "details": "something"}
            case "list":
                return [x + 5, x * 2]
            case "ChatDocument":
                return ChatDocument(
                    content=str(x + 5),
                    metadata=ChatDocMetaData(sender="Agent"),
                )
            case "pydantic":
                return SpecialResult(answer=x + 5)
            case "tool":
                # return tool, to be handled by sub-task
                return UberTool(x=x)
            case "final_tool":
                return FinalResultTool(
                    special=SpecialResult(answer=x + 5),  # explicitly declared
                    # arbitrary new fields that were not declared in the class...
                    extra_special=SpecialResult(answer=x + 10),
                    # ... does not need to be a Pydantic object
                    arbitrary_obj=dict(answer=x + 15),
                )
            case "agent_done":
                # pass on to parent, to handle with UberTool,
                # which is NOT enabled for this agent
                return AgentDoneTool(tools=[UberTool(x=x)])

    class UberTool(ToolMessage):
        request: str = "uber_tool"
        purpose: str = "to request the 'uber' transform of a  number <x>"
        x: int

        def handle(self) -> Any:
            return FinalResultTool(answer=self.x + 5)

    class CoolToolWithHandle(ToolMessage):
        request: str = "cool_tool"
        purpose: str = "to request the 'cool' transform of a  number <x>"

        x: int

        def handle(self) -> Any:
            return result_fn(self.x)

    class MyAgent(ChatAgent):
        def init_state(self) -> None:
            self.state: int = 100
            self.sender: str = ""

    class CoolToolWithResponse(ToolMessage):
        """To test that `response` handler works as expected,
        and is able to read and modify agent state.
        """

        request: str = "cool_tool"
        purpose: str = "to request the 'cool' transform of a  number <x>"

        x: int

        def response(self, agent: MyAgent) -> Any:
            agent.state += 1
            return result_fn(self.x)

    class CoolToolWithResponseDoc(ToolMessage):
        """
        To test that `response` handler works as expected,
        is able to read and modify agent state, and
        when using a `chat_doc` argument, and is able to read values from it.
        """

        request: str = "cool_tool"
        purpose: str = "to request the 'cool' transform of a  number <x>"

        x: int

        def response(self, agent: MyAgent, chat_doc: ChatDocument) -> Any:
            agent.state += 1
            agent.sender = chat_doc.metadata.sender
            return result_fn(self.x)

    match tool_handler:
        case "handle":
            tool_class = CoolToolWithHandle
        case "response":
            tool_class = CoolToolWithResponse
        case "response_with_doc":
            tool_class = CoolToolWithResponseDoc

    agent = MyAgent(
        ChatAgentConfig(
            name="Test",
            # no need for a real LLM, use a mock
            llm=MockLMConfig(
                # mock LLM generating a CoolTool variant
                response_fn=lambda x: tool_class(x=int(x)).json(),
            ),
        )
    )

    agent.enable_message(tool_class)
    # whether to terminate task on agent_response
    tool_result = result_type in ["final_tool", "agent_done", "tool"]
    task = Task(
        agent,
        interactive=False,
        # need to specify task done when result is not FinalResultTool
        done_if_response=[] if tool_result else [Entity.AGENT],
    )
    result = task.run("3")
    if tool_handler == "response":
        assert agent.state == 101
    if tool_handler == "response_with_doc":
        assert agent.state == 101
        assert agent.sender == "LLM"

    if not tool_result:
        # CoolTool handler returns a non-tool result containing 8, and
        # we terminate task on agent_response, via done_if_response,
        # so the result.content == 8
        assert "8" in result.content
    else:
        # When CoolTool handler returns a ToolMessage,
        # test that it is handled correctly by sub-task or a parent.

        another_agent = ChatAgent(
            ChatAgentConfig(
                name="Another",
                llm=MockLMConfig(response_fn=lambda x: x),  # pass thru
            )
        )
        another_agent.enable_message(UberTool)
        another_task = Task(another_agent, interactive=False)
        another_task.add_sub_task(task)
        result = another_task.run("3")

        if result_type == "final_tool":
            # task's CoolTool handler returns FinalResultTool
            # which short-circuits parent task and returns as a tool
            # in tool_messages list of the final result
            tool = result.tool_messages[0]
            assert isinstance(tool, FinalResultTool)
            assert isinstance(tool.special, SpecialResult)
            assert tool.special.answer == 8
            assert tool.extra_special.answer == 13
            assert tool.arbitrary_obj["answer"] == 18
        elif result_type == "agent_done":
            # inner task's CoolTool handler returns a DoneTool containing
            # UberTool, which is handled by the parent "another_agent"
            # which returns a FinalResultTool containing answer == 8
            tool = result.tool_messages[0]
            assert isinstance(tool, FinalResultTool)
            assert tool.answer == 8

            # Now disable parent agent's handling of UberTool
            another_agent.disable_message_handling(UberTool)
            # another_task = Task(another_agent, interactive=False)
            # another_task.add_sub_task(task)
            result = another_task.run("3")
            # parent task is unable to handle UberTool, so will stall
            # will stall and return None
            assert result is None
            another_agent.enable_message(UberTool)

        elif result_type == "tool":
            # inner Task CoolTool handler returns UberTool (with NO done signal),
            # which it is unable to handle, so stalls and returns None,
            # and so does parent another_task
            assert result is None

            # Now reverse it: make another_task a sub-task of task, and
            # test handling UberTool returned by task handler, by sub-task another_task
            another_task = Task(another_agent, interactive=False)
            # task = Task(agent, interactive=False)
            task.add_sub_task(another_task)
            result = task.run("3")
            tool = result.tool_messages[0]
            assert isinstance(tool, FinalResultTool)
            assert tool.answer == 8

            another_agent.disable_message_handling(UberTool)
            result = task.run("3")
            # subtask stalls, parent stalls, returns None
            assert result is None

@pytest.mark.parametrize("llm_tool", ["pair", "final_tool"])
@pytest.mark.parametrize("handler_result_type", ["agent_done", "final_tool"])
def test_llm_end_with_tool(handler_result_type:str, llm_tool:str):
    """
    Test that an LLM can indirectly trigger task-end, and return a Tool as result.
    """

    class Pair(BaseModel):
        a: int
        b: int

    class PairTool(ToolMessage):
        """Handle the LLM-generated tool, signal done or final-result and
        return it as the result."""

        request: str = "pair_tool"
        purpose: str = "to return a <pair> of numbers"
        pair: Pair

        def handle(self) -> Any:
            if handler_result_type == "final_tool":
                return FinalResultTool(result=self)
            else:
                return AgentDoneTool(tools=[self])

    class FinalResultPairTool(FinalResultTool):
        request:str = "final_result_pair_tool"
        purpose:str = "Present final result <pair>"
        pair: Pair

    final_result_pair_tool_name = FinalResultPairTool.default_value("request")

    class MyAgent(ChatAgent):
        def init_state(self) -> None:
            self.numbers: List[int] = []

        def user_response(
            self,
            msg: Optional[str | ChatDocument] = None,
        ) -> Optional[ChatDocument]:
            """Mock human user input: they start with 0, then increment by 1"""
            last_num = self.numbers[-1] if self.numbers else 0
            new_num = last_num + 1
            self.numbers.append(new_num)
            return self.create_user_response(content=str(new_num))

    pair_tool_name = PairTool.default_value("request")


    if llm_tool == "pair":
        # LLM generates just PairTool , to be handled by its tool handler
        system_message=f"""   
            Ask the user for their next number. 
            Once you have collected 2 distinct numbers, present these as a pair 
            using the TOOL: `{pair_tool_name}`.
            """
    else:
        system_message = f"""
            Ask the user for their next number. 
            Once you have collected 2 distinct numbers, present these as the 
            final result using the TOOL: `{final_result_pair_tool_name}`.
        """

    system_mess
    agent = MyAgent(
        ChatAgentConfig(
            name="MyAgent",
            system_message=system_message,
        )
    )
    if llm_tool == "pair":
        agent.enable_message(PairTool)
    else:
        agent.enable_message(FinalResultPairTool)

    # we are mocking user response, so need to set only_user_quits_root=False
    # so that the done signal (AgentDoneTool or FinalResultTool) actually end the task.
    task = Task(agent, interactive=True, only_user_quits_root=False)
    result = task.run()
    tool = result.tool_messages[0]
    if llm_tool == "pair":
        if handler_result_type == "final_tool":
            assert isinstance(tool, FinalResultTool)
            assert tool.result.pair.a == 1 and tool.result.pair.b == 2
        else:
            assert isinstance(tool, PairTool)
            assert tool.pair.a == 1 and tool.pair.b == 2
    else:
        assert isinstance(tool, FinalResultPairTool)
        assert tool.pair.a == 1 and tool.pair.b == 2
