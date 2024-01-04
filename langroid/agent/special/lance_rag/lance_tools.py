import logging

from langroid.agent.tool_message import ToolMessage

logger = logging.getLogger(__name__)


class QueryPlanTool(ToolMessage):
    request = "query_plan"  # the agent method name that handles this tool
    purpose = """
    Given a user's <original_query>, generate the components of a query plan:
    - <filter> condition if needed (or empty string if no filter is needed)
    - <query> - a possibly rephrased query that can be used to match the CONTENT
        of the documents (can be same as <original_query> if no rephrasing is needed)
    - <dataframe_calc> - a Pandas-dataframe calculation/aggregation string
        that can be used to calculate the answer 
        (or empty string if no calculation is needed).
    """
    original_query: str
    query: str
    filter: str
    dataframe_calc: str = ""
    result: str = ""


class QueryPlanFeedbackTool(ToolMessage):
    request = "query_plan_feedback"
    purpose = "To give <feedback> regarding the query plan."
    feedback: str
