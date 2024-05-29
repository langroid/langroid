from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import tools
    from . import description_extractors
    from . import populate_metadata
    from . import system_message
    from .tools import (
        RunQueryTool,
        GetTableNamesTool,
        GetTableSchemaTool,
        GetColumnDescriptionsTool,
    )
else:
    RunQueryTool = LazyLoad("langroid.agent.special.sql.utils.tools.RunQueryTool")
    GetTableNamesTool = LazyLoad(
        "langroid.agent.special.sql.utils.tools.GetTableNamesTool"
    )
    GetTableSchemaTool = LazyLoad(
        "langroid.agent.special.sql.utils.tools.GetTableSchemaTool"
    )

    GetColumnDescriptionsTool = LazyLoad(
        "langroid.agent.special.sql.utils.tools.GetColumnDescriptionsTool"
    )

    description_extractors = LazyLoad(
        "langroid.agent.special.sql.utils.description_extractors"
    )
    populate_metadata = LazyLoad("langroid.agent.special.sql.utils.populate_metadata")
    system_message = LazyLoad("langroid.agent.special.sql.utils.system_message")
    tools = LazyLoad("langroid.agent.special.sql.utils.tools")

__all__ = [
    "RunQueryTool",
    "GetTableNamesTool",
    "GetTableSchemaTool",
    "GetColumnDescriptionsTool",
    "description_extractors",
    "populate_metadata",
    "system_message",
    "tools",
]
