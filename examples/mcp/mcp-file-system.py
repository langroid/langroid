"""
Example: Expose local file-system operations via an in-memory FastMCP server.

Run like this:

uv run examples/mcp/mcp-file-system.py --model gpt-4.1-mini

Then ask your agent to list, write, or read files.
"""

import asyncio
import os

from fastmcp.server import FastMCP
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp import get_tool_async, mcp_tool
from pydantic import Field


def create_fs_mcp_server() -> FastMCP:
    """Return a FastMCP server exposing list/read/write file tools."""
    server = FastMCP("FsServer")

    @server.tool()
    def list_files(
        directory: str = Field(..., description="Directory path to list")
    ) -> list[str]:
        """List file names in the given directory."""
        try:
            return os.listdir(directory)
        except FileNotFoundError:
            return []

    @server.tool()
    def write_file(
        path: str = Field(..., description="Path to write to"),
        content: str = Field(..., description="Text content to write"),
    ) -> bool:
        """Write text to a file; return True on success."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    @server.tool()
    def read_file(
        path: str = Field(..., description="Path of a text file to read")
    ) -> str:
        """Read and return the content of a text file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    return server


# use decorator to create a Langroid ToolMessage with a custom handle_async method
@mcp_tool(create_fs_mcp_server(), "write_file")
class WriteFileTool(lr.ToolMessage):
    """Tool to write text to a file."""

    async def handle_async(self) -> str:
        """Invoke `write_file` and report the result."""
        ok = await self.call_tool_async()  # type: ignore
        return f"Wrote {self.path}: {ok}"


# use decorator to create a Langroid ToolMessage with a custom handle_async method
@mcp_tool(create_fs_mcp_server(), "read_file")
class ReadFileTool(lr.ToolMessage):
    """Tool to read the content of a text file."""

    async def handle_async(self) -> str:
        """Invoke `read_file` and return its contents."""
        text = await self.call_tool_async()  # type: ignore
        return text or ""


async def main(model: str = "") -> None:
    """
    Launch a ChatAgent that can list, write, and read files.

    Args:
    model: Optional LLM model name (defaults to gpt-4.1-mini).
    """
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=500,
                async_stream_quiet=False,
            ),
        )
    )

    # create ListFilesTool using the helper function get_tool_async
    ListFilesTool = await get_tool_async(create_fs_mcp_server(), "list_files")

    # enable all three tools
    agent.enable_message([ListFilesTool, WriteFileTool, ReadFileTool])

    # create a non-interactive task
    task = lr.Task(agent, interactive=False)

    # instruct the agent
    prompt = """
    1. List files in the current directory.
    2. Write a file 'note.txt' containing "Hello, MCP!".
    3. Read back 'note.txt'.
    """
    result = await task.run_async(prompt, turns=3)
    print(result.content)


if __name__ == "__main__":

    def _run(**kwargs: str) -> None:
        """Fire entry point to run the async main function."""
        asyncio.run(main(**kwargs))

    Fire(_run)
