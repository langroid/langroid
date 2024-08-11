import logging

from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    original_query: str = Field(..., description="The original query for reference")
    query: str = Field(..., description="A possibly NON-EMPTY rephrased query")
    filter: str = Field(
        "",
        description="Filter condition if needed (or empty if no filter is needed)",
    )
    dataframe_calc: str = Field(
        "", description="An optional Pandas-dataframe calculation/aggregation string"
    )


class QueryPlanTool(ToolMessage):
    request = "query_plan"  # the agent method name that handles this tool
    purpose = """
    Given a user's query, generate a query <plan> consisting of:
    - <original_query> - the original query for reference
    - <filter> condition if needed (or empty string if no filter is needed)
    - <query> - a possibly NON-EMPTY rephrased query that can be used to match the 
        CONTENT of the documents 
        (can be same as <original_query> if no rephrasing is needed)
    - <dataframe_calc> - a Pandas-dataframe calculation/aggregation string
        that can be used to calculate the answer 
        (or empty string if no calculation is needed).
    """
    plan: QueryPlan


class AnswerTool(ToolMessage):
    """Wrapper for answer from LanceDocChatAgent"""

    purpose: str = "To package the answer from LanceDocChatAgent"
    request: str = "answer_tool"
    answer: str


class QueryPlanAnswerTool(ToolMessage):
    request: str = "query_plan_answer"  # the agent method name that handles this tool
    purpose: str = """
    Assemble query <plan> and <answer>
    """
    plan: QueryPlan
    answer: str = Field(..., description="The answer received from the assistant")


class QueryPlanFeedbackTool(ToolMessage):
    request = "query_plan_feedback"
    purpose = """
    To give <feedback> regarding the query plan, 
    along with a <suggested_fix> if any (empty string if no fix is suggested).
    """
    feedback: str
    suggested_fix: str
