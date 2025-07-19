"""
Agent that uses a Tool to execute python code.

CAUTION - this is a security risk, as it allows arbitrary code execution.
This is a bare-bones example. For a real application, you would want to restrict
the code in various ways, e.g. by using a sandboxed environment, or by restricting
the modules that can be imported.

Run like this (leave model empty to use default GPT4o)

uv run examples/basic/python-code-exec-tool.py -m gpt4o-mini
"""

import contextlib
import io

from fire import Fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ResultTool
from pydantic import Field


def execute_code(code_string):
    """
    A minimal function to execute Python code and capture its output.

    Args:
        code_string: The Python code to execute

    Returns:
        Tuple of (output, local_variables)
    """
    # Create dictionary for local variables
    local_vars = {}

    # Capture stdout
    buffer = io.StringIO()

    # Execute code with stdout redirection
    with contextlib.redirect_stdout(buffer):
        try:
            exec(code_string, globals(), local_vars)
            success = True
        except Exception as e:
            print(f"Error: {str(e)}")
            success = False

    output = buffer.getvalue()
    return output, local_vars, success


class PyCodeTool(lr.ToolMessage):
    request: str = "py_code_tool"
    purpose: str = "To execute python <code> and return results"

    code: str = Field(
        ...,
        description="""
            Syntactically valid Python code that can be placed in file to 
            be run by the Python interpreter. MUST NOT CONTAIN any CODE-BLOCK
            delimiters like triple-backticks.
            """,
    )

    def handle(self):
        output, local_vars, success = execute_code(self.code)
        if success:
            print("Successfully ran code. Results:")
            print(output)
            print("Local variables:")
            print(local_vars)
        else:
            print("Failed to run code.")
        return ResultTool(output=output, local_vars=local_vars, success=success)


def main(model: str = ""):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Coder",
            llm=llm_config,
            # handle LLM non-tool msg
            handle_llm_no_tool=lambda msg: ResultTool(
                output=msg.content,
                success=True,
            ),
            system_message=f"""
            You are an expert python coder. When you get a user's message, 
            respond as follows:
            - if you think you need to run Python code,
                use the TOOL `{PyCodeTool.name()}` to perform the task.
            - otherwise simply respond to the user's message.
            """,
        )
    )
    agent.enable_message(PyCodeTool)
    # task specialized to return ResultTool
    # set restart to False to maintain conv history across `run` calls
    task = lr.Task(agent, interactive=False, restart=False)[ResultTool]

    while True:
        user_input = Prompt.ask("User")
        if user_input.lower() in ["x", "q"]:
            break
        result: ResultTool | None = task.run(user_input)
        if result is not None:
            # code was run; do something with the output if any
            if result.success:
                print("Output:", result.output)
            else:
                print("Code execution failed.")


if __name__ == "__main__":
    Fire(main)
