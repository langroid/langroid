"""
Use Langroid to set up a collaboration among three agents:

- Processor: needs to transform a list of positive numbers, does not know how to
apply the transformations, and sends out each number so that one of two
specialized agents apply the transformation. It is instructed to avoid getting a
negative number.
- EvenHandler only transforms even numbers, otherwise returns a negative number
- OddHandler only transforms odd numbers, otherwise returns a negative number

Since the Processor must avoid getting a negative number, it needs to
specify a recipient for each number it sends out,
using the `recipient_message` tool/function-call, where the `content` field
is the number it wants to send, and the `recipient` field is the name
of the intended recipient, either "EvenHandler" or "OddHandler".

This tool/function-call also has built-in mechanisms to remind the LLM
to specify a recipient if it forgets to do so.

Run as follows:

python3 examples/quick-start/two-agent-chat-num-router.py

For more explanation, see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/three-agent-chat-num-router/)
"""

import typer

import langroid as lr

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


def chat(tools: bool = False) -> None:
    config = lr.ChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(
            chat_model=lr.language_models.OpenAIChatModel.GPT4o,
        ),
        use_tools=tools,
        use_functions_api=not tools,
        vecdb=None,
    )
    processor_agent = lr.ChatAgent(config)
    processor_agent.enable_message(lr.agent.tools.RecipientTool)
    processor_task = lr.Task(
        processor_agent,
        name="Processor",
        system_message="""
        You will receive a list of numbers from me (the user).
        Your goal is to apply a transformation to each number.
        However you do not know how to do this transformation.
        You can take the help of two people to perform the 
        transformation.
        If the number is even, send it to EvenHandler,
        and if it is odd, send it to OddHandler.
        
        IMPORTANT: send the numbers ONE AT A TIME
        
        The handlers will transform the number and give you a new number.        
        If you send it to the wrong person, you will receive a negative value.
        Your aim is to never get a negative number, so you must 
        clearly specify who you are sending the number to.
        
        Once all numbers in the given list have been transformed, 
        say DONE and show me the result. 
        Start by asking me for the list of numbers.
        """,
        llm_delegate=True,
        single_round=False,
    )
    even_agent = lr.ChatAgent(config)
    even_task = lr.Task(
        even_agent,
        name="EvenHandler",
        system_message="""
        You will be given a number. 
        If it is even, divide by 2 and say the result, nothing else.
        If it is odd, say -10
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    odd_agent = lr.ChatAgent(config)
    odd_task = lr.Task(
        odd_agent,
        name="OddHandler",
        system_message="""
        You will be given a number n. 
        If it is odd, return (n*3+1), say nothing else. 
        If it is even, say -10
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    processor_task.add_sub_task([even_task, odd_task])
    processor_task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    tools: bool = typer.Option(
        False,
        "--tools",
        "-t",
        help="use langroid tools instead of OpenAI function-calling",
    ),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat(tools)


if __name__ == "__main__":
    app()
