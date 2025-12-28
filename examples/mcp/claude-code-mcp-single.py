"""
Enable a Langroid agent to use a SINGLE MCP Tool from 
Claude Code's MCP server.

Similar to claude-code-mcp.py but showing how to use a single tool, i.e.,
Claude-Code's special Grep tool that is built on ripgrep.

Run like this (omitting the `--model` argument will use the default gpt-5-mini):

    uv run examples/mcp/claude-code-mcp-single.py --model gpt-5-mini


"""

from fastmcp.client.transports import (
    StdioTransport,
)
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp import mcp_tool
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction


transport = StdioTransport(
    command="claude",
    args=["mcp", "serve"],
    env={},
)


# Illustrating how we can:
# - use the MCP tool decorator to create a Langroid ToolMessage subclass
# - override the handle_async() method to customize the output, sent to the LLM


@mcp_tool(transport, "Grep")
class GrepTool(lr.ToolMessage):
    async def handle_async(self):
        # Force a predictable, structured response shape from Grep so the
        # handler can parse and decide deterministically.
        # "content" mode returns matching lines along with numLines/numMatches.
        if hasattr(self, "output_mode"):
            self.output_mode = "content"

        # CODEX: Minimal, readable post‑processing — parse the structured JSON
        # and present plain text fields (no JSON) so the LLM can skim quickly.
        # The task will terminate on the LLM’s non‑tool reply due to
        # handle_llm_no_tool=Done.
        # Call the actual tool. Langroid returns a tuple (text, files). Unpack
        # the text payload for presentation.
        result = await self.call_tool_async()
        result_text, _files = result if isinstance(result, tuple) else (result, [])
        import json

        summary = None
        lines = None
        try:
            data = json.loads(result_text) if isinstance(result_text, str) else {}
            if isinstance(data, dict):
                mode = data.get("mode", "?")
                num_files = data.get("numFiles")
                filenames = data.get("filenames") or []
                num_lines = data.get("numLines")
                num_matches = data.get("numMatches")
                applied_limit = data.get("appliedLimit")
                applied_offset = data.get("appliedOffset")
                content_block = data.get("content", "")

                parts = [
                    f"mode: {mode}",
                    f"files matched: {num_files if num_files is not None else 0}",
                    (
                        "filenames: "
                        + (
                            ", ".join(filenames)
                            if isinstance(filenames, list) and filenames
                            else "(none)"
                        )
                    ),
                    (f"lines matched: {num_lines}" if num_lines is not None else None),
                    (
                        f"total matches: {num_matches}"
                        if num_matches is not None
                        else None
                    ),
                    (
                        f"applied limit: {applied_limit}"
                        if applied_limit is not None
                        else None
                    ),
                    (
                        f"applied offset: {applied_offset}"
                        if applied_offset is not None
                        else None
                    ),
                ]
                summary = "\n".join(p for p in parts if p)
                lines = str(content_block or "").rstrip()
        except Exception:
            pass

        if summary is not None:
            return f"""
            Grep summary (no JSON):
            {summary}

            Matching lines:
            {lines if lines else "(none)"}

            """
        else:
            # Fallback: show raw payload if parsing failed
            return f"""
            Grep result:
            {result_text}

            Answer the user's question with "yes" or "no" first, then briefly justify
            using the lines shown above.
            """


async def main(model: str = ""):
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # when the LLM responds without a tool, finish with that content
            # (prevents waiting for user input in non-interactive mode)
            handle_llm_no_tool=NonToolAction.DONE,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-5-mini",
                max_output_tokens=1000,
                # this defaults to True, but we set it to False so we can see output
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the grep tool
    agent.enable_message(GrepTool)
    task = lr.Task(agent, interactive=False)
    user_prompt = """
        Use your Grep MCP tool to check whether the pyproject.toml file in the current
        directory contains the string "hatch".
        """

    result = await task.run_async(user_prompt)
    assert "yes" in (result.content or "").lower()


if __name__ == "__main__":
    import asyncio

    def run_main(**kwargs) -> None:
        """Run the async main function with a proper event loop.

        Args:
            **kwargs: Keyword arguments to pass to the main function.
        """
        asyncio.run(main(**kwargs))

    Fire(run_main)
