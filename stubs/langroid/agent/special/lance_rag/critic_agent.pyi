from _typeshed import Incomplete

from langroid.agent.chat_agent import ChatAgent as ChatAgent
from langroid.agent.chat_document import ChatDocument as ChatDocument
from langroid.agent.special.lance_rag.query_planner_agent import (
    LanceQueryPlanAgentConfig as LanceQueryPlanAgentConfig,
)
from langroid.agent.special.lance_tools import (
    QueryPlanAnswerTool as QueryPlanAnswerTool,
)
from langroid.agent.special.lance_tools import (
    QueryPlanFeedbackTool as QueryPlanFeedbackTool,
)
from langroid.mytypes import Entity as Entity
from langroid.utils.constants import DONE as DONE
from langroid.utils.constants import NO_ANSWER as NO_ANSWER
from langroid.utils.constants import PASS as PASS

logger: Incomplete

class QueryPlanCriticConfig(LanceQueryPlanAgentConfig):
    name: str
    system_message: Incomplete

def plain_text_query_plan(msg: QueryPlanAnswerTool) -> str: ...

class QueryPlanCritic(ChatAgent):
    config: Incomplete
    def __init__(self, cfg: LanceQueryPlanAgentConfig) -> None: ...
    def query_plan_answer(self, msg: QueryPlanAnswerTool) -> str: ...
    def query_plan_feedback(self, msg: QueryPlanFeedbackTool) -> str: ...
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None: ...
