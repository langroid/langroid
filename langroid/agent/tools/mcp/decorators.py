from typing import Callable, Type

from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.mcp.fastmcp_client import (
    FastMCPServerSpec,
    get_tool,
)


def mcp_tool(
    server: FastMCPServerSpec, tool_name: str
) -> Callable[[Type[ToolMessage]], Type[ToolMessage]]:
    """Decorator: declare a ToolMessage class bound to a FastMCP tool.

    Usage:
        @mcp_tool("/path/to/server.py", "get_weather")
        class WeatherTool:
            def pretty(self) -> str:
                return f"Temp is {self.temperature}"

    The `server` may be a string/URL/FastMCP/ClientTransport, or a zero-arg
    callable returning one of those, e.g. `lambda: StdioTransport(...)`. Using a
    factory ensures a fresh transport per connection under fastmcp>=2.13.
    """

    def decorator(user_cls: Type[ToolMessage]) -> Type[ToolMessage]:
        # build the “real” ToolMessage subclass for this server/tool
        RealTool: Type[ToolMessage] = get_tool(server, tool_name)

        # copy user‐defined methods / attributes onto RealTool
        for name, attr in user_cls.__dict__.items():
            if name.startswith("__") and name.endswith("__"):
                continue
            setattr(RealTool, name, attr)

        # preserve the user’s original name if you like:
        RealTool.__name__ = user_cls.__name__
        return RealTool

    return decorator
