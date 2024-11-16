import typer
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from rich.prompt import Prompt
from langroid.utils.constants import DONE
from langroid.utils.logging import setup_colored_logging
from langroid.utils.configuration import set_global, Settings

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

    # set up tasks and subtasks
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
