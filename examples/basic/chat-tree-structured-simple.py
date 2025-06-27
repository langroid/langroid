"""
Simple example showing tree-structured computation, 
a variation of `examples/basic/chat-tree.py` which uses strict output formatting
to reliably wrap calls to agents in standard Python functions, allowing
explicit control over control flow.

The task consists of performing this calculation for a given input number n:

def Main(n):
    if n is odd:
        return (3*n+1) + n
    else:
        If n is divisible by 10:
            return n/10 + n
        else:
            return n/2 + n

Each step is performed by an LLM call, and strict output formatting ensures that
a valid typed response is returned (rather than a string which requires another
LLM call to interpret).

We evaluate the conditions with a `condition_agent` which is given an integer and
a condition and return a Boolean and evaluate the transformations of `n` with
a `transformation_agent` which is given an integer and a transformation rule
and returns the transformed integer.

Finally, we add the result with the original `n` using an `adder_agent` which
illustrates strict output usage in `Task`s.

For more details on structured outputs, see the notes at
https://langroid.github.io/langroid/notes/structured-output/.
"""

import typer
from rich.prompt import Prompt

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


def chat() -> int:
    condition_agent = ChatAgent(
        ChatAgentConfig(
            system_message="""
            You will be provided with a condition and a
            number; your goal is to determine whether
            that number satisfies the condition.

            Respond in JSON format, with `value` set
            to the result.
            """,
            output_format=bool,
        )
    )
    transformation_agent = ChatAgent(
        ChatAgentConfig(
            system_message="""
            You will be provided with a number and an
            transformation of the number to perform.

            Respond in JSON format, with `value` set
            to the result.
            """,
            output_format=int,
        )
    )

    def check_condition(n: int, condition: str) -> bool:
        output = condition_agent.llm_response_forget(
            f"""
            Number: {n}
            Condition: {condition}
            """
        )
        return condition_agent.from_ChatDocument(output, bool)  # type: ignore

    def apply_transformation(n: int, transformation: str) -> int:
        output = transformation_agent.llm_response_forget(
            f"""
            Number: {n}
            Transformation: {transformation}
            """
        )
        return transformation_agent.from_ChatDocument(output, int)  # type: ignore

    num = int(Prompt.ask("Enter a number"))
    is_even = check_condition(num, "The number is even.")

    if is_even:
        is_divisible_by_10 = check_condition(num, "The number is divisible by 10.")

        if is_divisible_by_10:
            to_adder = apply_transformation(num, "n/10 where the number is n.")
        else:
            to_adder = apply_transformation(num, "n/2 where the number is n.")
    else:
        to_adder = apply_transformation(num, "3n+1 where the number is n.")

    class AddNumTool(ToolMessage):
        request = "add_num"
        purpose = "Add <number> to the original number, return the result"
        number: int

        def handle(self) -> str:
            total = num + self.number
            return f"{DONE} {total}"

    # We could also have the agent output a the call in a single step and handle
    # it ourselves (or apply it immediately)
    adder_agent = ChatAgent(
        ChatAgentConfig(
            system_message="""
            You will be given a number n.
            You have to add it to the original number and return the result.
            You do not know the original number, so you must use the 
            `add_num` tool/function for this. 
            """,
            output_format=AddNumTool,
        )
    )
    adder_agent.enable_message(AddNumTool)
    adder_task = Task(adder_agent, interactive=False, name="Adder")

    # compute the final output value
    return adder_task[int].run(str(to_adder))  # type: ignore


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
