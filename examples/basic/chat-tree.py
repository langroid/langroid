"""
Simple example showing tree-structured computation
where each node in the tree is handled by a separate agent.

This task consists of performing this calculation for a given input number n:

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
"""

import typer
from rich.prompt import Prompt

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import DONE
from langroid.utils.globals import GlobalState
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


class MyGlobalState(GlobalState):
    number: int | None = None


class AskNumTool(ToolMessage):
    request = "ask_num"
    purpose = "Ask user for the initial number"

    def handle(self) -> str:
        """
        This is a stateless tool (i.e. does not use any Agent member vars), so we can
        define the handler right here, instead of defining an `ask_num`
        method in the agent.
        """
        num = Prompt.ask("Enter a number")
        # record this in global state, so other agents can access it
        MyGlobalState.set_values(number=num)
        return str(num)


class AddNumTool(ToolMessage):
    request = "add_num"
    purpose = "Add <number> to the original number, return the result"
    number: int

    def handle(self) -> str:
        """
        This is a stateless tool (i.e. does not use any Agent member vars), so we can
        define the handler right here, instead of defining an `add_num`
        method in the agent.
        """
        return str(int(MyGlobalState.get_value("number")) + int(self.number))


def chat(model: str = "") -> None:
    config = ChatAgentConfig(
        llm=OpenAIGPTConfig(
            chat_model=model or OpenAIChatModel.GPT4o,
        ),
        vecdb=None,
    )

    main_agent = ChatAgent(config)
    main_task = Task(
        main_agent,
        name="Main",
        interactive=False,
        system_message="""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        In this case simply write the <number>, say nothing else.
        
        RESULT Message format: RESULT <number>
        In this case simply say "DONE <number>", e.g.:
        DONE 19

        To start off, ask the user for the initial number, 
        using the `ask_num` tool/function.
        """,
    )

    # Handles only even numbers
    even_agent = ChatAgent(config)
    even_task = Task(
        even_agent,
        name="Even",
        interactive=False,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if the <number> is odd, say '{DONE}'
        - otherwise, simply write the <number>, say nothing else.
        
        RESULT Message format: RESULT <number>
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    # handles only even numbers ending in Zero
    evenz_agent = ChatAgent(config)
    evenz_task = Task(
        evenz_agent,
        name="EvenZ",
        interactive=False,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if <number> n is even AND divisible by 10, compute n/10 and pass it on,
        - otherwise, say '{DONE}'
        
        RESULT Message format: RESULT <number>
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    # Handles only even numbers NOT ending in Zero
    even_nz_agent = ChatAgent(config)
    even_nz_task = Task(
        even_nz_agent,
        name="EvenNZ",
        interactive=False,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if <number> n is even AND NOT divisible by 10, compute n/2 and pass it on,
        - otherwise, say '{DONE}'
        
        RESULT Message format: RESULT <number>
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    # Handles only odd numbers
    odd_agent = ChatAgent(config)
    odd_task = Task(
        odd_agent,
        name="Odd",
        interactive=False,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if <number> n is odd, compute n*3+1 and write it.
        - otherwise, say '{DONE}'

        RESULT Message format: RESULT <number>        
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    adder_agent = ChatAgent(config)
    adder_task = Task(
        adder_agent,
        name="Adder",
        interactive=False,
        system_message="""
        You will be given a number n.
        You have to add it to the original number and return the result.
        You do not know the original number, so you must use the 
        `add_num` tool/function for this. 
        When you receive the result, say "DONE RESULT <result>", e.g.
        DONE RESULT 19
        """,
    )

    # set up tasks and subtasks
    main_task.add_sub_task([even_task, odd_task])
    even_task.add_sub_task([evenz_task, even_nz_task])
    evenz_task.add_sub_task(adder_task)
    even_nz_task.add_sub_task(adder_task)
    odd_task.add_sub_task(adder_task)

    # set up the tools
    main_agent.enable_message(AskNumTool)
    adder_agent.enable_message(AddNumTool)

    # start the chat
    main_task.run()


@app.command()
def main(
    model: str = typer.Option("", "--model", "-m", help="model to use"),
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
    chat(model)


if __name__ == "__main__":
    app()
