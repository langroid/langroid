import asyncio
import logging
import traceback
from typing import Optional

import fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from examples.code_execution.docker_code_executor import (
    CodeBlock,
    CodeResult,
    DockerCodeExecutor,
    DockerCodeExecutorConfig,
)
from langroid.agent.tools.orchestration import ResultTool
from langroid.pydantic_v1 import Field, PrivateAttr
from langroid.utils.configuration import Settings, set_global

set_global(Settings(debug=True))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DockerPyCodeTool(lr.ToolMessage):
    request: str = "docker_py_code_tool"
    purpose: str = (
        """Execute Python code in Docker. Provide ONLY raw Python code in the code field"""
    )
    code: str = Field(..., description="Raw Python code to execute (no markdown)")

    _executor: DockerCodeExecutor = PrivateAttr()

    @classmethod
    def set_executor(
        cls,
        executor: DockerCodeExecutor,
        config: Optional[DockerCodeExecutorConfig] = None,
    ):
        if config is None:
            DockerPyCodeTool._executor = executor
        else:
            DockerPyCodeTool._executor = DockerCodeExecutor(config=config)

    async def handle_async(self):
        """
        Handles the execution of the Python code within the Docker container asynchronously.
        Returns the formatted output string directly.
        """
        if not hasattr(self, "_executor") or self._executor is None:
            logger.error("Executor not set on DockerPyCodeTool instance!")
            return ResultTool(
                output="Execution error: Executor not initialized. Exit Code: -1",
                success=False,
            )

        if not self.code.strip():
            logger.warning("Code is empty, returning error string")
            return ResultTool(
                output="Execution error: Empty code provided. Exit Code: 1",
                success=False,
            )

        try:
            code_block = CodeBlock(code=self.code, language="python")
            logger.info(
                f"Executing code in Docker (executor ID: {id(self._executor)}): {self.code[:100]}..."
            )
            if not self._executor._is_running:
                logger.info("Executor wasn't running, attempting to start...")
                await self._executor.start()

            result: CodeResult = await self._executor.execute_code_blocks([code_block])

            # Combine output and exit code into the string to be returned
            output_str = f"Exit Code: {result.exit_code}\nOutput:\n{result.output}"

            logger.info(f"Returning execution result string: {output_str[:100]}...")

            return ResultTool(output=output_str, success=True)

        except Exception as e:
            logger.error(f"Exception during Docker execution: {e}")
            tb = traceback.format_exc()
            logger.error(f"Traceback:\n{tb}")
            error_str = f"Execution error: {str(e)}\n{tb}. Exit Code: -1"
            return ResultTool(output=error_str, success=False)


async def main(model: str = ""):
    # Create a DockerCodeExecutorConfig object with explicit parameters
    executor_config = DockerCodeExecutorConfig(
        image="python:3.11-slim",  # Docker image to use
        container_name="my-langroid-executor",  # Optional container name
        timeout=1000,  # Execution timeout in seconds
        work_dir=None,  # Host directory for code execution (defaults to temp dir)
        auto_remove=True,  # Remove container on stop
        stop_container=True,  # Stop container on exit
    )
    executor = DockerCodeExecutor(config=executor_config)
    await executor.start()

    llm_config = lm.OpenAIGPTConfig(chat_model="gemini/gemini-2.0-flash")
    try:
        # Set executor on tool class
        DockerPyCodeTool.set_executor(executor)

        # Configure agent
        agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                name="CodeExecutor",
                llm=llm_config,
                system_message=f"""
            You are an expert python coder. When you get a user's message,
            respond as follows:
            - If you need to perform a calculation or execute Python code,
              use the TOOL `{ DockerPyCodeTool.name()}` to perform the task.
            - After using the tool, respond to the user with the result,
              explaining what the result means in the context of their original
              request.
            - If you don't need to use the tool, simply respond to the user's message.
            """,
                use_tools=True,
                use_functions_api=False,
            )
        )
        agent.enable_message(DockerPyCodeTool)

        # Correctly define the task to return ResultTool and set restart=False
        # Also specify that the task should return a ResultTool
        task = lr.Task(agent, interactive=False, restart=False, inf_loop_wait_factor=1)[
            ResultTool
        ]

        print("\n--- Agent Ready ---")
        print("Example requests (remember to ask for the tool to be used):")
        print("- Calculate fibonacci(100)")
        print("- What is 15 factorial?")
        print("- Print hello world")

        while True:
            user_input = Prompt.ask("User")
            if user_input.lower() in ["x", "q"]:
                break

            result: ResultTool | None = await task.run_async(user_input)

            if result is not None:
                if result.success:
                    print("Output:", result.output)
                else:
                    print("Code execution failed.")
                    if result.output:
                        print("Error details:", result.output)
            else:
                print("Task finished without a final result.")

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        await executor.stop()
        print("\n--- Docker Executor Stopped ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(lambda model="": asyncio.run(main(model=model)))
