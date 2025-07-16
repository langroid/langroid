"""
Other tests for Task are in test_chat_agent.py
"""

from typing import Any

import pytest

import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool, ResultTool
from langroid.pydantic_v1 import BaseModel
from langroid.utils.constants import DONE


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
            msg = Pair(x=2, y=4)  # gets converted to str via .model_dump_json()

    if final_result_type == "agent_done_tool":
        # Run twice: ensure default is not overriden
        for _ in range(2):
            result = task.run(msg)
            # default result -> Optional[ChatDocument]
            assert isinstance(result, lr.ChatDocument)
            # in the `content_any` field of the final ChatDocument,
            # an arbitrary type can be stored, as returned by AgentDoneTool(content=...)
            assert isinstance(result.content_any, DetailedAnswer)
            assert result.content_any.answer == 6
            assert result.content_any.comment != ""

            result = task[DetailedAnswer].run(msg)
            assert isinstance(result, DetailedAnswer)
            assert result.answer == 6
            assert result.comment != ""

        # Overridden return type takes precedence
        result = task[float].run(msg, return_type=DetailedAnswer)
        assert isinstance(result, DetailedAnswer)
        assert result.answer == 6
        assert result.comment != ""

        # Test default return type
        result = lr.Task(
            agent=agent,
            interactive=False,
            default_return_type=DetailedAnswer,
        ).run(msg)
        assert isinstance(result, DetailedAnswer)
        assert result.answer == 6
        assert result.comment != ""

    else:
        # default result -> Optional[ChatDocument]
        result = task.run(msg)
        tools = agent.get_tool_messages(result)
        assert isinstance(tools[0], ResultTool)
        assert tools[0].answer == 6

        # Test overriden return type
        result = task[str].run(msg, return_type=ResultTool)
        assert isinstance(result, ResultTool)
        assert result.answer == 6

        result = task[ResultTool].run(msg)
        assert isinstance(result, ResultTool)
        assert result.answer == 6

        result = task[list[ResultTool]].run(msg)
        assert isinstance(result, list) and isinstance(result[0], ResultTool)
        assert result[0].answer == 6

        result = task[ToolMessage].run(msg)
        assert isinstance(result, ResultTool)
        assert result.answer == 6

        result = task[int].run(msg)
        assert result == 6

        # check handling of invalid return type:
        # receive None when strict recovery is disabled
        agent.disable_strict = True
        result = task[Pair].run(msg)
        assert result is None
        agent.disable_strict = False

        # check we can return a Pydantic model
        result = task[DetailedAnswer].run(msg)
        assert isinstance(result, DetailedAnswer)
        assert result.answer == 6
        assert result.comment != ""

        # check we can return a dictionary
        result = task[dict[str, Any]].run(msg)
        assert isinstance(result, dict)
        assert result["answer"] == 6
        assert result["comment"] != ""

        # Test default return type
        result = lr.Task(
            agent=agent,
            interactive=False,
            default_return_type=dict[str, Any],
        ).run(msg)
        assert isinstance(result, dict)
        assert result["answer"] == 6
        assert result["comment"] != ""

        # Test we can set desired return type when creating task, using [...] syntax
        task = lr.Task(agent=agent, interactive=False)[ResultTool]
        result = task.run(msg)
        assert isinstance(result, ResultTool)
        assert result.answer == 6


def test_strict_recovery():
    """Tests strict JSON mode recovery for `Task`s with a `return_type`."""

    def collatz(n: int) -> int:
        if (n % 2) == 0:
            return n // 2

        return 3 * n + 1

    def collatz_sequence(n: int) -> list[int]:
        sequence = [n]

        while n != 1:
            n = collatz(n)
            sequence.append(n)

        return sequence

    class CollatzTool(lr.ToolMessage):
        request: str = "collatz"
        purpose: str = "To compute the value following `n` in a Collatz sequence."
        n: int

        def handle(self):
            return str(collatz(self.n))

    class CollatzSequence(BaseModel):
        sequence: list[int]

    agent = lr.ChatAgent()
    agent.enable_message(CollatzTool)
    task = lr.Task(
        system_message=f"""
        You will be provided with an integer (call it `n`);
        your goal is to compute the Collatz sequence
        starting at `n`. Do this by calling the `CollatzTool`
        tool/function on each subsequent value in the sequence,
        until the result becomes 1.

        Once it does, tell me the sequence of values and say {DONE}.
        """,
        interactive=False,
        erase_substeps=True,
        default_return_type=CollatzSequence,
    )

    def is_correct(n: int) -> bool:
        result = task.run(str(n))
        return isinstance(
            result, CollatzSequence
        ) and result.sequence == collatz_sequence(n)

    for n in range(2, 5):
        assert is_correct(n)


@pytest.mark.asyncio
async def test_strict_recovery_async():
    """Tests strict JSON mode recovery for `Task`s with a `return_type`."""

    def collatz(n: int) -> int:
        if (n % 2) == 0:
            return n // 2

        return 3 * n + 1

    def collatz_sequence(n: int) -> list[int]:
        sequence = [n]

        while n != 1:
            n = collatz(n)
            sequence.append(n)

        return sequence

    class CollatzTool(lr.ToolMessage):
        request: str = "collatz"
        purpose: str = "To compute the value following `n` in a Collatz sequence."
        n: int

        def handle(self):
            return str(collatz(self.n))

    class CollatzSequence(BaseModel):
        sequence: list[int]

    agent = lr.ChatAgent()
    agent.enable_message(CollatzTool)
    task = lr.Task(
        system_message=f"""
        You will be provided with an integer (call it `n`);
        your goal is to compute the Collatz sequence
        starting at `n`. Do this by calling the `CollatzTool`
        tool/function on each subsequent value in the sequence,
        until the result becomes 1.

        Once it does, tell me the sequence of values and say {DONE}.
        """,
        interactive=False,
        erase_substeps=True,
        default_return_type=CollatzSequence,
    )

    async def is_correct(n: int) -> bool:
        result = await task.run_async(str(n))
        return isinstance(
            result, CollatzSequence
        ) and result.sequence == collatz_sequence(n)

    for n in range(2, 5):
        assert await is_correct(n)
