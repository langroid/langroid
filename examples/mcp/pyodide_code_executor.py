"""
Simple example of using the Pyodide MCP server.
    https://github.com/pydantic/pydantic-ai/tree/main/mcp-run-python

Before running make sure you have deno installed
    https://docs.deno.com/runtime/getting_started/installation/

Run like this:

    uv run examples/mcp/pyodide_code_executor.py --model gpt-4.1-mini

"""

from fastmcp.client.transports import StdioTransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp import mcp_tool
from langroid.agent.tools.orchestration import ResultTool
from langroid.mytypes import NonToolAction

RUN_ONCE: bool = True # terminate task on first result?

deno_transport = StdioTransport(
    command="deno",
    args=[
        "run",
        "-N",
        "-R=node_modules",
        "-W=node_modules",
        "--node-modules-dir=auto",
        "jsr:@pydantic/mcp-run-python",
        "stdio",
    ],
)

# Illustrating how we can:
# - use the MCP tool decorator to create a Langroid ToolMessage subclass
# - override the handle_async() method to customize the output, sent to the LLM

class MyResult(ResultTool):
    answer: str

@mcp_tool(deno_transport, "run_python_code")
class PythonCodeExecutor(lr.ToolMessage):
    async def handle_async(self):
        result: str = await self.call_tool_async()
        if RUN_ONCE:
            # terminate task with this result
            return MyResult(answer=result)
        else:
            # this result goes to LLM, and loop with user continues
            return f"""
            <CodeResult>
            {result} 
            </CodeResult>
            """


async def main(model: str = ""):
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=1000,
                # this defaults to True, but we set it to False so we can see output
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the PythonCodeExecutor tool
    agent.enable_message(PythonCodeExecutor)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    if RUN_ONCE:
        task = lr.Task(agent, interactive=False)[MyResult]
        result: MyResult|None = await task.run_async()
        print("Final answer is: ", result.answer)
    else:
        task = lr.Task(agent, interactive=False)
        await task.run_async()


if __name__ == "__main__":
    import asyncio

    def run_main(**kwargs) -> None:
        """Run the async main function with a proper event loop.

        Args:
            **kwargs: Keyword arguments to pass to the main function.
        """
        asyncio.run(main(**kwargs))

    Fire(run_main)
