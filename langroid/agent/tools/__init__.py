from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import google_search_tool
    from . import recipient_tool
    from .google_search_tool import GoogleSearchTool
    from .recipient_tool import AddRecipientTool, RecipientTool
else:
    GoogleSearchTool = LazyLoad(
        "langroid.agent.tools.google_search_tool.GoogleSearchTool"
    )
    AddRecipientTool = LazyLoad("langroid.agent.tools.recipient_tool.AddRecipientTool")
    RecipientTool = LazyLoad("langroid.agent.tools.recipient_tool.RecipientTool")

    google_search_tool = LazyLoad("langroid.agent.tools.google_search_tool")
    recipient_tool = LazyLoad("langroid.agent.tools.recipient_tool")

__all__ = [
    "GoogleSearchTool",
    "AddRecipientTool",
    "RecipientTool",
    "google_search_tool",
    "recipient_tool",
]
