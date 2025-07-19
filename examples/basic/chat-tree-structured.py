"""
Simple example showing tree-structured computation
where each node in the tree is handled by a separate agent.
A variation of `examples/basic/chat-tree.py` which uses strict output formatting
and agent logic to enforce the behavior specified in the prompts.

See the use of `set_output_format()` in ConditionalAgent.

The task consists of performing this calculation for a given input number n:

def Main(n):
    if n is odd:
        return (3*n+1) + n
    else:
        If n is divisible by 10:
            return n/10 + n
        else:
            return n/2 + n

To make this "interesting", we represent this computation hierarchically,
in the form of this tree:

Main
- Odd
    - Adder
- Even
    - EvenZ
        - Adder
    - EvenNZ
        - Adder

For a full write-up on the design considerations, see the documentation page on
Hiearchical Agent Computations at https://langroid.github.io/langroid/examples/agent-tree/

For more details on structured outputs, see the notes at
https://langroid.github.io/langroid/notes/structured-output/.
"""

import typer
from rich.prompt import Prompt

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE
from langroid.utils.globals import GlobalState
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


class MyGlobalState(GlobalState):
    number: int | None = None


class AskNumTool(ToolMessage):
    request: str = "ask_num"
    purpose: str = "Ask user for the initial number"

    def handle(self) -> str:
        """
        This is a stateless tool (i.e. does not use any Agent member vars), so we can
        define the handler right here, instead of defining an `ask_num`
        method in the agent.
        """
        num = int(Prompt.ask("Enter a number"))
        # record this in global state, so other agents can access it
        MyGlobalState.set_values(number=num)
        return str(num)


class AddNumTool(ToolMessage):
    request: str = "add_num"
    purpose: str = "Add <number> to the original number, return the result"
    number: int

    def handle(self) -> AgentDoneTool:
        """
        This is a stateless tool (i.e. does not use any Agent member vars), so we can
        define the handler right here, instead of defining an `add_num`
        method in the agent.
        """
        total = MyGlobalState.get_value("number") + self.number
        return AgentDoneTool(
            tools=[ResultTool(result=total)],
        )


class MatchTool(ToolMessage):
    request: str = "match"
    purpose: str = "To express whether the input number matches your condition."
    matches: bool


class ResultTool(ToolMessage):
    request: str = "result"
    purpose: str = (
        "To express the result of your transformation applied to the input number."
    )
    result: int


class ConditionalAgentConfig(ChatAgentConfig):
    top_level: bool = False


class ConditionalAgent(ChatAgent):
    def __init__(self, config: ConditionalAgentConfig = ConditionalAgentConfig()):
        super().__init__(config)
        self.config: ConditionalAgentConfig = config  # type: ignore
        # Should the next request be treated as self-generated?
        self.generated_request: bool = False

        if self.config.top_level:
            # We always begin by requesting a number from the user
            self.set_output_format(AskNumTool)
            self.enable_message(AskNumTool)
            self.enable_message(ResultTool, handle=True, use=False)
        else:
            self.enable_message([MatchTool, ResultTool])
            # We always begin by checking whether the number matches the agent's condiditon
            self.set_output_format(MatchTool)

    def ask_num(self, msg: AskNumTool) -> str:
        self.set_output_format(None)
        return msg.handle()

    def match(self, msg: MatchTool) -> str:
        if not msg.matches:
            return DONE

        # The agent must next return the transformed number
        self.set_output_format(ResultTool)
        self.generated_request = True
        return "Now, return the input number, after applying your transformation."

    def result(self, msg: ResultTool) -> str | ChatDocument | AgentDoneTool:
        if self.config.top_level:
            self.set_output_format(AskNumTool)
            # Return the answer if we are the top-level task
            return f"{DONE} {msg.result}"
        elif self.generated_request:
            self.generated_request = False
            return self.create_llm_response(
                content=str(msg.result),
            )
        else:
            self.set_output_format(MatchTool)

        # Propogate the result up if we are done
        return AgentDoneTool(
            tools=[msg],
        )


def chat() -> None:
    main_task = Task(
        ConditionalAgent(
            ConditionalAgentConfig(
                top_level=True,
            )
        ),
        interactive=False,
        name="Main",
        system_message="""
        You will ask the user for a number with the `ask_num` tool; you should respond with exactly that number,
        say nothing else.
        """,
    )

    prompt_format = """
        You will receive a number; you should first check whether that number
        matches your condition.

        Condition: {condition}

        If so, you should respond with a transformed version of the number:

        Transformation: {transformation}
        """

    even_task = Task(
        ConditionalAgent(),
        interactive=False,
        name="Even",
        system_message=prompt_format.format(
            condition="The number is even.",
            transformation="Nothing, return the number you were provided.",
        ),
    )
    evenz_task = Task(
        ConditionalAgent(),
        interactive=False,
        name="EvenZ",
        system_message=prompt_format.format(
            condition="The number is divisible by 10.",
            transformation="Return n/10 where n is the provided number.",
        ),
    )
    even_nz_task = Task(
        ConditionalAgent(),
        interactive=False,
        name="EvenNZ",
        system_message=prompt_format.format(
            condition="The number is not divisible by 10.",
            transformation="Return n/2 where n is the provided number.",
        ),
    )
    odd_task = Task(
        ConditionalAgent(),
        interactive=False,
        name="Odd",
        system_message=prompt_format.format(
            condition="The number is odd.",
            transformation="Return n*3 + 1",
        ),
    )

    adder_agent = ChatAgent()
    adder_agent.enable_message(AddNumTool)
    adder_task = Task(
        # ensure that the agent calls the tool:
        # agent[T] is a copy of agent which always outputs values of type T
        adder_agent[AddNumTool],
        name="Adder",
        interactive=False,
        system_message="""
        You will be given a number n.
        You have to add it to the original number and return the result.
        You do not know the original number, so you must use the 
        `add_num` tool/function for this. 
        """,
    )

    # set up tasks and subtasks
    main_task.add_sub_task([even_task, odd_task])
    even_task.add_sub_task([evenz_task, even_nz_task])
    evenz_task.add_sub_task(adder_task)
    even_nz_task.add_sub_task(adder_task)
    odd_task.add_sub_task(adder_task)

    # start the chat
    main_task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()
