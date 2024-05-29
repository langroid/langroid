from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import query_planner_agent
    from . import critic_agent
    from . import lance_rag_task
else:
    query_planner_agent = LazyLoad(
        "langroid.agent.special.lance_rag.query_planner_agent"
    )
    critic_agent = LazyLoad("langroid.agent.special.lance_rag.critic_agent")
    lance_rag_task = LazyLoad("langroid.agent.special.lance_rag.lance_rag_task")

__all__ = [
    "query_planner_agent",
    "critic_agent",
    "lance_rag_task",
]
