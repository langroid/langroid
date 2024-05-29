from _typeshed import Incomplete

from langroid.agent.special.lance_doc_chat_agent import (
    LanceDocChatAgent as LanceDocChatAgent,
)
from langroid.agent.special.lance_rag.critic_agent import (
    QueryPlanCritic as QueryPlanCritic,
)
from langroid.agent.special.lance_rag.critic_agent import (
    QueryPlanCriticConfig as QueryPlanCriticConfig,
)
from langroid.agent.special.lance_rag.query_planner_agent import (
    LanceQueryPlanAgent as LanceQueryPlanAgent,
)
from langroid.agent.special.lance_rag.query_planner_agent import (
    LanceQueryPlanAgentConfig as LanceQueryPlanAgentConfig,
)
from langroid.agent.task import Task as Task
from langroid.mytypes import Entity as Entity

logger: Incomplete

class LanceRAGTaskCreator:
    @staticmethod
    def new(agent: LanceDocChatAgent, interactive: bool = True) -> Task: ...
