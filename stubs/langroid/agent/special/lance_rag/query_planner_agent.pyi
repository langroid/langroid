from _typeshed import Incomplete

from langroid.agent.chat_agent import (
    ChatAgent as ChatAgent,
)
from langroid.agent.chat_agent import (
    ChatAgentConfig as ChatAgentConfig,
)
from langroid.agent.chat_document import ChatDocument as ChatDocument
from langroid.agent.special.lance_tools import (
    QueryPlan as QueryPlan,
)
from langroid.agent.special.lance_tools import (
    QueryPlanAnswerTool as QueryPlanAnswerTool,
)
from langroid.agent.special.lance_tools import (
    QueryPlanFeedbackTool as QueryPlanFeedbackTool,
)
from langroid.agent.special.lance_tools import (
    QueryPlanTool as QueryPlanTool,
)
from langroid.utils.constants import (
    DONE as DONE,
)
from langroid.utils.constants import (
    NO_ANSWER as NO_ANSWER,
)
from langroid.utils.constants import (
    PASS_TO as PASS_TO,
)

logger: Incomplete

class LanceQueryPlanAgentConfig(ChatAgentConfig):
    name: str
    critic_name: str
    doc_agent_name: str
    doc_schema: str
    use_tools: bool
    max_retries: int
    use_functions_api: bool
    system_message: Incomplete
    def set_system_message(self) -> None: ...

class LanceQueryPlanAgent(ChatAgent):
    config: Incomplete
    curr_query_plan: Incomplete
    n_retries: int
    result: str
    def __init__(self, config: LanceQueryPlanAgentConfig) -> None: ...
    def query_plan(self, msg: QueryPlanTool) -> str: ...
    def query_plan_feedback(self, msg: QueryPlanFeedbackTool) -> str: ...
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None: ...
