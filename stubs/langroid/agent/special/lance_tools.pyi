from _typeshed import Incomplete
from pydantic import BaseModel

from langroid.agent.tool_message import ToolMessage as ToolMessage

logger: Incomplete

class QueryPlan(BaseModel):
    original_query: str
    query: str
    filter: str
    dataframe_calc: str

class QueryPlanTool(ToolMessage):
    request: str
    purpose: str
    plan: QueryPlan

class QueryPlanAnswerTool(ToolMessage):
    request: str
    purpose: str
    plan: QueryPlan
    answer: str

class QueryPlanFeedbackTool(ToolMessage):
    request: str
    purpose: str
    feedback: str
    suggested_fix: str
