import logging

from pydantic import BaseModel

from langroid.agent.tool_message import ToolMessage

logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    original_query: str
    query: str
    filter: str
    dataframe_calc: str = ""


class QueryPlanTool(ToolMessage):
    request = "query_plan"  # the agent method name that handles this tool
    purpose = """
    Given a user's query, generate a query <plan> consisting of:
    - <original_query> - the original query for reference
    - <filter> condition if needed (or empty string if no filter is needed)
    - <query> - a possibly rephrased query that can be used to match the CONTENT
        of the documents (can be same as <original_query> if no rephrasing is needed)
    - <dataframe_calc> - a Pandas-dataframe calculation/aggregation string
        that can be used to calculate the answer 
        (or empty string if no calculation is needed).
    """
    plan: QueryPlan


class QueryPlanAnswerTool(ToolMessage):
    request = "query_plan_answer"  # the agent method name that handles this tool
    purpose = """
    Assemble query <plan> and <answer>
    """
    plan: QueryPlan
    answer: str


class QueryPlanFeedbackTool(ToolMessage):
    request = "query_plan_feedback"
    purpose = """
    To give <feedback> regarding the query plan, 
    along with a <suggested_fix> if any (empty string if no fix is suggested).
    """
    feedback: str
    suggested_fix: str
