"""
Other tests for Task are in test_chat_agent.py
"""

from typing import Any, Dict, List

import pytest

import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool, ResultTool
from langroid.pydantic_v1 import BaseModel


@pytest.mark.parametrize(
    "input_type",
    ["GenPair", "Pair", "str", "int", "list", "dict"],
)
@pytest.mark.parametrize(
    "pair_tool_handler_return_type",
    [
        "Pair",
        "str",
        "list",
        "dict",
    ],
)
@pytest.mark.parametrize("final_result_type", ["agent_done_tool", "result_tool"])
def test_task_in_out_types(
    input_type: str,
    pair_tool_handler_return_type: str,
    final_result_type: str,
):
    """
    Test that we can have:

    result: TypeOut = task.run(input: TypeIn, return_type: TypeOut)

    i.e., task.run() can take a variety of input types and return desired output type
    """

    # TODO
    # - add more input types, like list, dict etc
    # - add more handler-result types, e.g. GenPair tool can return other than str,
    #   e.g. list of 2 nums, pydantic Pair object etc
    # - AgentDoneTool should accept content of any type, which we can stick into
    # content_any field etc. Maybe same thing for DoneTool
    #  - have a good way to check for compound types inside
    #  ToolMessage.find_value_of_type, e.g. list, dict etc
    # - carefully define content_any in various places

    class Pair(BaseModel):
        x: int
        y: int

    class DetailedAnswer(BaseModel):
        comment: str
        answer: int

    class CoolTool(lr.ToolMessage):
        request: str = "cool_tool"
        purpose: str = "to request the Cool Transform of a number <pair>"

        pair: Pair

        def handle(self) -> ResultTool:
            match final_result_type:
                case "result_tool":
                    return ResultTool(
                        answer=self.pair.x + self.pair.y,  # integer result
                        details=DetailedAnswer(  # Pydantic model result
                            comment="The CoolTransform is just the sum of the numbers",
                            answer=self.pair.x + self.pair.y,
                        ),
                        dictionary=dict(
                            comment="The CoolTransform is just the sum of the numbers",
                            answer=self.pair.x + self.pair.y,
                        ),
                    )
                case "agent_done_tool":
                    return AgentDoneTool(
                        content=DetailedAnswer(
                            comment="The CoolTransform is just the sum of the numbers",
                            answer=self.pair.x + self.pair.y,
                        )
                    )

    cool_tool_name = CoolTool.default_value("request")

    class GenPairTool(lr.ToolMessage):
        request: str = "input_tool"
        purpose: str = "to generate a number-pair from an integer <x>"

        x: int

        def handle(self) -> Any:
            match pair_tool_handler_return_type:
                case "str":
                    return f"Here is a pair of numbers: {self.x-1}, {self.x+1}"
                case "list":
                    return [self.x - 1, self.x + 1]
                case "dict":
                    return dict(first=self.x - 1, second=self.x + 1)
                case "Pair":
                    return Pair(x=self.x - 1, y=self.x + 1)

    gen_pair_tool_name = GenPairTool.default_value("request")

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="MyAgent",
            system_message=f"""
            When you receive a PAIR of numbers, request the Cool Transform of the pair,
            using the TOOL: `{cool_tool_name}`
            
            When you receive a SINGLE number, generate a PAIR of numbers from it,
            using the TOOL: `{gen_pair_tool_name}`
            """,
        )
    )
    agent.enable_message([CoolTool, GenPairTool])

    task = lr.Task(agent=agent, interactive=False)

    match input_type:
        case "str":
            msg = "Get the Cool Transform of the numbers: 2, 4"
        case "int":
            msg = 3
        case "list":
            msg = [2, 4]
        case "dict":
            msg = dict(first=2, second=4)
        case "GenPair":
            msg = GenPairTool(x=3)  # agent handler generates a Pair obj, list, etc.
        case "Pair":
            msg = Pair(x=2, y=4)  # gets converted to str via .json()

    if final_result_type == "agent_done_tool":
        result = task.run(msg)
        # default result -> Optional[ChatDocument]
        assert isinstance(result, lr.ChatDocument)
        # in the `content_any` field of the final ChatDocument,
        # an arbitrary type can be stored, as returned by AgentDoneTool(content=...)
        assert isinstance(result.content_any, DetailedAnswer)
        assert result.content_any.answer == 6
        assert result.content_any.comment != ""

        result = task.run(msg, return_type=DetailedAnswer)
        assert isinstance(result, DetailedAnswer)
        assert result.answer == 6
        assert result.comment != ""

    else:
        # default result -> Optional[ChatDocument]
        result = task.run(msg)
        tools = agent.get_tool_messages(result)
        assert isinstance(tools[0], ResultTool)
        assert tools[0].answer == 6

        result = task.run(msg, return_type=ResultTool)
        assert isinstance(result, ResultTool)
        assert result.answer == 6

        result = task.run(msg, return_type=List[ResultTool])
        assert isinstance(result, list) and isinstance(result[0], ResultTool)
        assert result[0].answer == 6

        result = task.run(msg, return_type=ToolMessage)
        assert isinstance(result, ResultTool)
        assert result.answer == 6

        result = task.run(msg, return_type=int)
        assert result == 6

        # check handling of invalid return type: receive None
        result = task.run(msg, return_type=Pair)
        assert result is None

        # check we can return a Pydantic model
        result = task.run(msg, return_type=DetailedAnswer)
        assert isinstance(result, DetailedAnswer)
        assert result.answer == 6
        assert result.comment != ""

        # check we can return a dictionary
        result = task.run(msg, return_type=Dict[str, Any])
        assert isinstance(result, dict)
        assert result["answer"] == 6
        assert result["comment"] != ""
