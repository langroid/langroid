"""
Simple example of using the GitMCP server to "chat" about a GitHub repository.

https://github.com/idosal/git-mcp

The server offers several tools, and we can enable ALL of them to be used
by a Langroid agent.

Run like this (-m model optional; defaults to gpt-4.1-mini):

    uv run examples/mcp/gitmcp.py -m ollama/qwen2.5-coder:32b

"""

from textwrap import dedent
from typing import List

from fastmcp.client.transports import SSETransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.agent.tools.orchestration import SendTool
from langroid.pydantic_v1 import Field


def get_gitmcp_url() -> str:
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    import re

    short_pattern = re.compile(r"^([^/]+)/([^/]+)$")
    url_pattern = re.compile(
        r"^(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+)(?:\.git)?/?$"
    )

    while True:
        user_input = Prompt.ask(
            "[bold blue]Enter the GitHub repository (owner/repo or full URL)"
        ).strip()
        m = short_pattern.match(user_input)
        if m:
            owner, repo = m.groups()
        else:
            m = url_pattern.match(user_input)
            if m:
                owner, repo = m.groups()
            else:
                console.print(
                    "[red]Invalid format. Please enter 'owner/repo' or a full GitHub URL."
                )
                continue
        break

    github_url = f"https://github.com/{owner}/{repo}"
    console.print(f"Full GitHub URL set to [green]{github_url}[/]")

    gitmcp_url = f"https://gitmcp.io/{owner}/{repo}"
    console.print(f"GitMCP URL set to [green]{gitmcp_url}[/]")
    return gitmcp_url


class SendUserTool(SendTool):
    request: str = "send_user"
    purpose: str = "Send <content> to user"
    to: str = "user"
    content: str = Field(
        ...,
        description="""
        Message to send to user, typically answer to user's request,
        or a clarification question to the user, if user's task/question
        is not completely clear.
        """,
    )


async def main(model: str = ""):

    gitmcp_url = get_gitmcp_url()

    transport = SSETransport(
        url=gitmcp_url,
    )
    all_tools: List[lr.ToolMessage] = await get_tools_async(transport)

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool="You FORGOT to use one of your TOOLs!",
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=10_000,
                async_stream_quiet=False,
            ),
            system_message=dedent(
                f"""
                Make best use of any of the TOOLs available to you,
                to answer the user's questions.
                To communicate with the User, you MUST use
                the TOOL `{SendUserTool.name()}` - typically this would
                be to either send the user your answer to their query/request,
                or to ask the user a clarification question, if the user's request
                is not completely clear.
                """
            ),
        )
    )

    # enable the agent to use all tools
    agent.enable_message(all_tools + [SendUserTool])
    # configure task to NOT recognize string-based signals like DONE,
    # since those could occur in the retrieved text!
    task_cfg = lr.TaskConfig(recognize_string_signals=False)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool

    task = lr.Task(agent, config=task_cfg, interactive=False)
    await task.run_async(
        "Based on the TOOLs available to you, greet the user and"
        "tell them what kinds of help you can provide."
    )


if __name__ == "__main__":
    Fire(main)
